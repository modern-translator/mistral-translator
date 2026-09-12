import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import DOMPurify from "dompurify";
import {
  AlertCircle, BookOpen, Check, CheckCircle2, ChevronRight, Copy, Download,
  Eye, FileText, Globe, HelpCircle, ImagePlus, Languages, Loader2, Minus,
  RotateCcw, Settings, ShieldCheck, Sparkles, Upload, X, Zap
} from "lucide-react";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.8.69/pdf.worker.min.mjs";

// DEFAULT ENGINE: intentionally independent from the manual Flash cascade.
const GEMINI_MODELS = ["gemini-3.5-flash-lite"];
const FLASH_MODELS = [
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3-flash-preview"
];
const MAX_REQUESTS_PER_KEY_MODEL = 500;
const MAX_REQUESTS_PER_KEY_MODEL_FLASH = 20;
const REQUEST_INTERVAL_MS = 5000;
const FLASH_REQUEST_INTERVAL_MS = 15000;
const TOTAL_API_KEY_SLOTS = 10;
const SESSION_STORAGE_KEY = "translator_session_v3";
const API_KEYS_STORAGE_KEY = "translator_api_keys";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const textOnly = (html = "") => {
  const el = document.createElement("div");
  el.innerHTML = html;
  return (el.textContent || "").replace(/\s+/g, " ").trim();
};
const countWords = (value = "") => value.trim() ? value.trim().split(/\s+/).length : 0;
const cleanModelHtml = (value = "") => value.replace(/```(?:html)?/gi, "").trim();

function safeSanitize(html = "") {
  return DOMPurify.sanitize(html, {
    ADD_TAGS: ["input", "button"],
    ADD_ATTR: [
      "class", "style", "accept", "type", "data-diagram-id", "data-action",
      "data-page-id", "alt", "dir", "role", "aria-label"
    ],
    FORBID_TAGS: ["script", "iframe", "object", "embed", "form"]
  });
}

// GROUP 1: retry-safe, independently throttled fetch. Retry-After is honored
// for transient quota responses without merging the two engine clocks.
async function fetchWithRetry(url, options, timeoutMs, acquireSlot) {
  const delays = [1000, 2000, 4000, 8000, 16000];
  let lastError;
  for (let attempt = 0; attempt <= delays.length; attempt += 1) {
    await acquireSlot();
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      window.clearTimeout(timeoutId);
      if (!response.ok) {
        const error = new Error(`API Error ${response.status}: ${await response.text().catch(() => response.statusText)}`);
        error.status = response.status;
        const retryAfter = response.headers.get("Retry-After");
        if (retryAfter) {
          const seconds = Number(retryAfter);
          error.retryAfterMs = Number.isFinite(seconds) ? seconds * 1000 : 0;
        }
        throw error;
      }
      return await response.json();
    } catch (error) {
      window.clearTimeout(timeoutId);
      lastError = error;
      if ([400, 403, 429].includes(error.status)) throw error;
      if (attempt < delays.length) await sleep(Math.max(delays[attempt], error.retryAfterMs || 0));
    }
  }
  throw lastError;
}

function makeRotation() {
  return { keyIdx: 0, modelIdx: 0, count: 0, modelFailCount: 0, deadKeys: new Set() };
}

function updatePage(prev, pageId, updater) {
  return prev.map((page) => page.id === pageId ? updater(page) : page);
}

// GROUP 2: geometry-first PDF metadata. These helpers are deliberately
// non-destructive: rawText remains available as the compatibility source.
// GROUP 2: conservative column detection. A large horizontal gap creates a
// column only when the evidence is clear; ambiguous pages stay single-column.
function detectColumnRanges(items, pageWidth) {
  if (items.length < 8) return [{ start: 0, end: pageWidth, index: 0 }];
  const starts = [...new Set(items.map((item) => Math.round(item.x)))]
    .sort((a, b) => a - b);
  const gaps = starts.map((value, index) => ({
    value, gap: index ? value - starts[index - 1] : 0
  })).filter((item) => item.gap > pageWidth * 0.18);
  if (!gaps.length || gaps.length > 3) return [{ start: 0, end: pageWidth, index: 0 }];
  const split = gaps[0].value;
  return [
    { start: 0, end: split, index: 0 },
    { start: split, end: pageWidth, index: 1 }
  ];
}

function groupLines(items, tolerance = 3, columnRanges = [{ start: 0, end: Infinity, index: 0 }]) {
  const lines = [];
  [...items].sort((a, b) => b.y - a.y || a.x - b.x).forEach((item) => {
    const column = columnRanges.find((range) => item.x >= range.start && item.x < range.end)?.index || 0;
    let line = lines.find((candidate) => candidate.column === column && Math.abs(candidate.y - item.y) <= tolerance);
    if (!line) {
      line = { id: `line-${lines.length + 1}`, x: item.x, y: item.y, width: 0, height: item.height, column, items: [] };
      lines.push(line);
    }
    line.items.push(item);
    line.x = Math.min(line.x, item.x);
    line.width = Math.max(line.width, item.x + item.width - line.x);
    line.height = Math.max(line.height, item.height);
  });
  return lines
    .sort((a, b) => a.column - b.column || b.y - a.y)
    .map((line) => ({
      ...line,
      items: line.items.sort((a, b) => a.x - b.x),
      text: line.items.map((item) => item.str).join(" ").trim()
    }));
}

function classifyBlocks(lines, pageId, pageHeight) {
  const nonEmpty = lines.filter((line) => line.text);
  const blocks = [];
  let current = null;
  nonEmpty.forEach((line, index) => {
    const isSmall = line.height < (nonEmpty[0]?.height || line.height) * 0.78;
    const nearBottom = line.y < pageHeight * 0.18;
    const looksNumbered = /^(\(?\d+[\].)]|[•●▪-])\s/.test(line.text);
    const looksHeading = line.text.length < 100 && (line.height > (nonEmpty[0]?.height || line.height) * 1.12 || /^[A-Z\u0980-\u09ff\s\d:,-]{3,}$/.test(line.text));
    const looksQuote = /^[“"«(]/.test(line.text) || (line.x > pageHeight * 0.12 && line.text.length > 60);
    const looksCitation = /\b(Vol\.|pp?\.|পৃষ্ঠা|حَدِيث|قال|narrated|ibid)\b/i.test(line.text);
    const looksTable = /\s{3,}|\t/.test(line.text);
    const type = nearBottom && isSmall ? "footnote"
      : looksNumbered ? "list_item"
      : looksTable ? "table_candidate"
      : looksCitation ? "citation"
      : looksQuote ? "quote"
      : looksHeading ? "heading"
      : "paragraph";
    const canJoin = current && current.type === type && Math.abs(current.y - line.y) < pageHeight * 0.08;
    if (canJoin) {
      current.text += ` ${line.text}`;
      current.lines.push(line.id);
      current.height = Math.max(current.height, line.height);
    } else {
      current = {
        id: `page-${pageId}-block-${String(blocks.length + 1).padStart(2, "0")}`,
        type, text: line.text, lines: [line.id], x: line.x, y: line.y,
        width: line.width, height: line.height, confidence: type === "paragraph" ? "high" : "medium"
      };
      blocks.push(current);
    }
  });
  return blocks;
}

function detectFootnoteWarnings(blocks) {
  const footnotes = blocks.filter((block) => block.type === "footnote");
  const body = blocks.filter((block) => block.type !== "footnote").map((block) => block.text).join(" ");
  const markers = [...body.matchAll(/(?:^|\s)[¹²³⁴⁵⁶⁷⁸⁹*](?=\s|$)|(?:^|\s)\d+[.)](?=\s)/g)].map((m) => m[0].trim());
  const warnings = [];
  if (markers.length > new Set(markers).size) warnings.push("Duplicate footnote markers detected.");
  if (markers.length && !footnotes.length) warnings.push("Body markers have no confidently detected footnote block.");
  if (footnotes.length && !markers.length) warnings.push("Footnote-like text has no matching body marker.");
  return { markers, warnings };
}

// GROUP 2: repeated page furniture and cross-page context are metadata only.
// They never alter source text or authorize sentence completion.
function finalizeDocumentMetadata(pages) {
  const frequency = new Map();
  pages.forEach((page) => {
    const candidates = [page.blocks[0]?.text, page.blocks[page.blocks.length - 1]?.text]
      .filter((value) => value && value.length > 3);
    candidates.forEach((value) => frequency.set(value, (frequency.get(value) || 0) + 1));
  });
  const repeated = new Set([...frequency.entries()].filter(([, count]) => count > 1).map(([value]) => value));
  return pages.map((page, index) => {
    const blocks = page.blocks.map((block, blockIndex) => {
      const isPageNumber = blockIndex === page.blocks.length - 1 && /^\s*[\[(]?\d{1,5}[\])\s.-]*$/.test(block.text);
      const isHeader = blockIndex === 0 && repeated.has(block.text);
      const isFooter = blockIndex === page.blocks.length - 1 && repeated.has(block.text);
      return {
        ...block,
        type: isPageNumber ? "page_number" : isHeader ? "header" : isFooter ? "footer" : block.type
      };
    });
    const previous = pages[index - 1];
    const last = blocks[blocks.length - 1];
    const firstNext = pages[index + 1]?.blocks?.[0];
    const continuation = Boolean(
      last && firstNext &&
      !/[.!?؟۔:؛]$/.test(last.text.trim()) &&
      /^[a-z\u0980-\u09ff\u0600-\u06ff]/i.test(firstNext.text.trim())
    );
    return {
      ...page,
      blocks,
      repeatedMetadata: {
        repeatedHeaderFooterText: [...repeated].filter((value) => blocks.some((block) => block.text === value)),
        previousPageId: previous?.id ?? null,
        nextPageId: pages[index + 1]?.id ?? null,
        likelyContinuationFromPrevious: Boolean(previous?.blocks?.length && !/[.!?؟۔:؛]$/.test(previous.blocks.at(-1).text.trim()) && blocks[0]?.text),
        likelyContinuationToNext: continuation
      }
    };
  });
}

function normalizePage(page) {
  return {
    ...page,
    meta: page.meta || { partName: `Page ${page.id}` },
    content: page.content || { rawText: "" },
    lines: page.lines || [],
    blocks: page.blocks || [],
    geometry: page.geometry || { width: 800, height: 1100, aspectRatio: 800 / 1100, orientation: "portrait" },
    visuals: page.visuals || [],
    columns: page.columns || [{ start: 0, end: page.geometry?.width || 800, index: 0 }],
    repeatedMetadata: page.repeatedMetadata || {
      repeatedHeaderFooterText: [], previousPageId: null, nextPageId: null,
      likelyContinuationFromPrevious: false, likelyContinuationToNext: false
    },
    warnings: page.warnings || [],
    glossaryTerms: page.glossaryTerms || [],
    translationStatus: page.translationStatus || "idle",
    banglaHtml: page.banglaHtml || "",
    originalBanglaHtml: page.originalBanglaHtml || "",
    engineProduced: page.engineProduced || null,
    verificationStatus: page.verificationStatus || "not-run"
  };
}

function validateTranslation(html, sourceText) {
  const output = textOnly(html);
  const sourceWords = countWords(sourceText);
  const outputWords = countWords(output);
  const warnings = [];
  if (!output) return { ok: false, warnings: ["The model returned empty output."] };
  if (sourceWords > 45 && outputWords < Math.max(8, Math.floor(sourceWords * 0.12))) {
    warnings.push("Output is unusually short compared with the source.");
  }
  if (sourceWords > 80 && outputWords < 20) warnings.push("Translation may be incomplete.");
  if (/^[\s.,:;!?-]+$/.test(output)) warnings.push("Output contains no readable content.");
  if (/<(?:html|body|head)\b/i.test(html)) warnings.push("Model returned document-level markup; review structure.");
  return { ok: true, warnings };
}

function ensureVisualIdentity(html, pageId) {
  let index = 0;
  return html.replace(/<div\b([^>]*class=["'][^"']*diagram-placeholder[^"']*["'][^>]*)>/gi, (full, attrs) => {
    index += 1;
    if (/data-diagram-id\s*=/.test(attrs)) return full;
    return `<div${attrs} data-diagram-id="page-${pageId}-visual-${index}">`;
  });
}

function glossaryFromTranslation(page, html) {
  const sourceTerms = (page.blocks || []).filter((block) => ["heading", "citation", "religious_quote"].includes(block.type)).slice(0, 6);
  const translatedParts = [...textOnly(html).split(/[.!?।]/)].filter(Boolean);
  return sourceTerms.slice(0, translatedParts.length).map((term, index) => ({
    source: term.text.slice(0, 80),
    approved: translatedParts[index].trim().slice(0, 100)
  })).filter((term) => term.approved);
}

const App = () => {
  const [fileData, setFileData] = useState(null);
  const [parsedSections, setParsedSections] = useState([]);
  const [activeSectionId, setActiveSectionId] = useState(null);
  const [sourceLangMode, setSourceLangMode] = useState("auto");
  const [glossary, setGlossary] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [parseProgress, setParseProgress] = useState(0);
  const [isTranslatingAll, setIsTranslatingAll] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [apiKeys, setApiKeys] = useState(() => {
    try {
      const raw = sessionStorage.getItem(API_KEYS_STORAGE_KEY) || localStorage.getItem(API_KEYS_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) && parsed.length === TOTAL_API_KEY_SLOTS
        ? parsed : Array(TOTAL_API_KEY_SLOTS).fill("");
    } catch { return Array(TOTAL_API_KEY_SLOTS).fill(""); }
  });
  const [rememberApiKey, setRememberApiKey] = useState(() => !!localStorage.getItem(API_KEYS_STORAGE_KEY));
  const [showSettings, setShowSettings] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [copiedId, setCopiedId] = useState("");
  const [reviewMode, setReviewMode] = useState("target");
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [activeSourceImage, setActiveSourceImage] = useState("");

  const pdfDocRef = useRef(null);
  const pageImageCacheRef = useRef(new Map());
  const jobsRef = useRef(new Map());
  const rotationRef = useRef(makeRotation());
  const flashRotationRef = useRef(makeRotation());
  const throttleRef = useRef(Promise.resolve());
  const flashThrottleRef = useRef(Promise.resolve());
  const lastRequestRef = useRef(0);
  const lastFlashRequestRef = useRef(0);
  const restoredRef = useRef(false);
  const fileInputRef = useRef(null);

  const setPage = useCallback((pageId, updater) => {
    setParsedSections((prev) => updatePage(prev, pageId, updater));
  }, []);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "null");
      if (saved?.parsedSections?.length) {
        setFileData(saved.fileData || null);
        setParsedSections(saved.parsedSections.map(normalizePage));
        setActiveSectionId(saved.activeSectionId ?? saved.parsedSections[0].id);
        setSourceLangMode(saved.sourceLangMode || "auto");
        setGlossary(saved.glossary || []);
        setSuccessMsg("Restored your previous in-progress session.");
      }
    } catch (error) {
      console.warn("Could not restore previous session:", error);
    } finally {
      restoredRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (!restoredRef.current) return;
    try {
      if (parsedSections.length) {
        localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
          fileData, parsedSections, activeSectionId, sourceLangMode, glossary
        }));
      } else {
        localStorage.removeItem(SESSION_STORAGE_KEY);
      }
    } catch (error) {
      console.warn("Session storage is full; translation state remains available in memory.", error);
    }
  }, [fileData, parsedSections, activeSectionId, sourceLangMode, glossary]);

  useEffect(() => {
    let mounted = true;
    if (!activeSectionId) return undefined;
    const page = parsedSections.find((item) => item.id === activeSectionId);
    if (!page?.isPdf) return undefined;
    (async () => {
      const image = await extractPageImageBase64(activeSectionId);
      if (mounted) setActiveSourceImage(image ? `data:image/jpeg;base64,${image}` : "");
    })();
    return () => { mounted = false; };
  }, [activeSectionId, parsedSections]);

  const scheduleSlot = useCallback((flash = false) => {
    const queueRef = flash ? flashThrottleRef : throttleRef;
    const lastRef = flash ? lastFlashRequestRef : lastRequestRef;
    const interval = flash ? FLASH_REQUEST_INTERVAL_MS : REQUEST_INTERVAL_MS;
    const slot = queueRef.current.then(async () => {
      const wait = interval - (Date.now() - lastRef.current);
      if (wait > 0) await sleep(wait);
      lastRef.current = Date.now();
    });
    queueRef.current = slot.catch(() => {});
    return slot;
  }, []);

  const callGemini = useCallback(async (parts, useFlash = false) => {
    const activeKeys = apiKeys.map((key) => key.trim()).filter(Boolean);
    if (!activeKeys.length) throw new Error("NO_API_KEY");
    const models = useFlash ? FLASH_MODELS : GEMINI_MODELS;
    const rotation = useFlash ? flashRotationRef.current : rotationRef.current;
    const maxPerModel = useFlash ? MAX_REQUESTS_PER_KEY_MODEL_FLASH : MAX_REQUESTS_PER_KEY_MODEL;
    if (rotation.keyIdx >= activeKeys.length) rotation.keyIdx = 0;
    const maxAttempts = activeKeys.length * models.length * (useFlash ? 2 : 1);
    let attempts = 0;
    while (attempts < maxAttempts) {
      while (rotation.deadKeys.has(rotation.keyIdx) && rotation.deadKeys.size < activeKeys.length) {
        rotation.keyIdx = (rotation.keyIdx + 1) % activeKeys.length;
        rotation.modelIdx = 0;
        rotation.count = 0;
        rotation.modelFailCount = 0;
      }
      if (rotation.deadKeys.size >= activeKeys.length) throw new Error("ALL_KEYS_EXHAUSTED");
      const key = activeKeys[rotation.keyIdx];
      const model = models[rotation.modelIdx];
      try {
        const data = await fetchWithRetry(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ contents: [{ parts }] }) },
          60000,
          () => scheduleSlot(useFlash)
        );
        rotation.count += 1;
        // GROUP 1: Flash quota failures are reset only after a successful request.
        rotation.modelFailCount = 0;
        if (rotation.count >= maxPerModel) {
          rotation.count = 0;
          rotation.modelIdx = (rotation.modelIdx + 1) % models.length;
          if (rotation.modelIdx === 0) rotation.keyIdx = (rotation.keyIdx + 1) % activeKeys.length;
        }
        return data;
      } catch (error) {
        const isQuota = error.status === 429 || /RESOURCE_EXHAUSTED|quota/i.test(error.message || "");
        const isBadKey = error.status === 400 || error.status === 403 || /API_KEY_INVALID|PERMISSION_DENIED/i.test(error.message || "");
        if (isQuota) {
          if (useFlash) {
            // GROUP 1: one transient 429 does not abandon a Flash model.
            rotation.modelFailCount = (rotation.modelFailCount || 0) + 1;
            if (rotation.modelFailCount < 2) {
              attempts += 1;
              if (error.retryAfterMs) await sleep(error.retryAfterMs);
              continue;
            }
          }
          rotation.modelFailCount = 0;
          rotation.modelIdx += 1;
          rotation.count = 0;
          if (rotation.modelIdx >= models.length) {
            rotation.modelIdx = 0;
            rotation.keyIdx = (rotation.keyIdx + 1) % activeKeys.length;
          }
          attempts += 1;
          continue;
        }
        if (isBadKey) {
          rotation.deadKeys.add(rotation.keyIdx);
          rotation.keyIdx = (rotation.keyIdx + 1) % activeKeys.length;
          rotation.modelIdx = 0;
          rotation.count = 0;
          rotation.modelFailCount = 0;
          attempts += 1;
          continue;
        }
        throw error;
      }
    }
    throw new Error("ALL_KEYS_EXHAUSTED");
  }, [apiKeys, scheduleSlot]);

  async function extractPageImageBase64(pageId) {
    if (pageImageCacheRef.current.has(pageId)) return pageImageCacheRef.current.get(pageId);
    if (!pdfDocRef.current) return null;
    try {
      const page = await pdfDocRef.current.getPage(pageId);
      const viewport = page.getViewport({ scale: 2.5 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d", { alpha: false });
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: ctx, viewport }).promise;
      const result = canvas.toDataURL("image/jpeg", 0.9).split(",")[1];
      pageImageCacheRef.current.set(pageId, result);
      return result;
    } catch (error) {
      console.warn("Could not render page image", pageId, error);
      return null;
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setErrorMsg(""); setSuccessMsg("");
    if (!file.name.toLowerCase().endsWith(".pdf")) return setErrorMsg("Invalid format. Please upload a PDF document.");
    if (file.size > 100 * 1024 * 1024) return setErrorMsg("File size exceeds 100MB limit.");
    setIsParsing(true); setParseProgress(0);
    try {
      const data = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data }).promise;
      pdfDocRef.current = pdf;
      pageImageCacheRef.current.clear();
      const pages = [];
      for (let id = 1; id <= pdf.numPages; id += 1) {
        const page = await pdf.getPage(id);
        const viewport = page.getViewport({ scale: 1 });
        const content = await page.getTextContent();
        const items = content.items.filter((item) => item.str !== undefined).map((item, index) => ({
          str: item.str, x: item.transform?.[4] || 0, y: item.transform?.[5] || 0,
          width: item.width || 0, height: item.height || Math.abs(item.transform?.[3] || 12),
          transform: item.transform || [], fontName: item.fontName || "", fontSize: Math.abs(item.transform?.[0] || 12),
          sourceIndex: index
        }));
        const columnRanges = detectColumnRanges(items, viewport.width);
        const lines = groupLines(items, 3, columnRanges);
        const blocks = classifyBlocks(lines, id, viewport.height);
        const footnote = detectFootnoteWarnings(blocks);
        const pageText = lines.map((line) => line.text).filter(Boolean).join(" ");
        // GROUP 2: image operators become stable, non-destructive visual
        // metadata. Placeholder HTML remains model-driven and independent.
        const operatorList = await page.getOperatorList();
        const imageOps = new Set([
          pdfjsLib.OPS.paintImageMaskXObject,
          pdfjsLib.OPS.paintImageXObject,
          pdfjsLib.OPS.paintSolidColorImageMask
        ]);
        const visuals = operatorList.fn
          .map((fn, index) => imageOps.has(fn) ? {
            assetId: `page-${id}-visual-${index + 1}`,
            type: "image",
            pageId: id,
            x: 0, y: 0, width: viewport.width, height: viewport.height,
            description: "PDF image operator; exact bounds unavailable from text geometry"
          } : null)
          .filter(Boolean);
        pages.push(normalizePage({
          id,
          meta: { partName: `Page ${id}` },
          content: { rawText: pageText, textItems: items },
          lines, blocks, columns: columnRanges,
          geometry: {
            width: viewport.width, height: viewport.height,
            aspectRatio: viewport.width / viewport.height,
            orientation: viewport.width >= viewport.height ? "landscape" : "portrait"
          },
          visuals,
          warnings: footnote.warnings,
          isPdf: true
        }));
        if (id % 3 === 0 || id === pdf.numPages) {
          setParseProgress(Math.round((id / pdf.numPages) * 100));
          await sleep(0);
        }
      }
      const finalizedPages = finalizeDocumentMetadata(pages);
      setFileData({ name: file.name, size: (file.size / 1048576).toFixed(2) });
      setParsedSections(finalizedPages); setActiveSectionId(finalizedPages[0]?.id || null);
      setSuccessMsg(`Document loaded: ${pages.length} page${pages.length === 1 ? "" : "s"}.`);
    } catch (error) {
      console.error(error);
      setErrorMsg("Error reading the file. Please verify the document.");
    } finally {
      setIsParsing(false);
    }
  };

  const buildSharedContext = (page) => {
    const blockContext = page.blocks?.length
      ? page.blocks.map((block) => `[${block.id} | ${block.type}] ${block.text}`).join("\n")
      : page.content.rawText;
    const glossaryContext = glossary.slice(-20).map((term) => `${term.source} → ${term.approved}`).join("; ");
    const previous = parsedSections.find((item) => item.id === page.id - 1);
    const next = parsedSections.find((item) => item.id === page.id + 1);
    return {
      blockContext,
      glossaryContext,
      continuationContext: [
        previous?.blocks?.find((block) => block.type === "heading")?.text,
        next?.blocks?.find((block) => block.type === "heading")?.text
      ].filter(Boolean).join(" | ") || "none"
    };
  };

  // GROUP 3: one prompt builder is shared by default, Retry, Flash, and Verify.
  const buildTranslationPrompt = (page) => {
    const context = buildSharedContext(page);
    const lang = sourceLangMode === "ar_ur" ? "Arabic / Urdu" : sourceLangMode === "en" ? "English" : "Arabic, Urdu, or English";
    return `Translate this ${lang} PDF page into faithful Bangla styled HTML.
The current page is the only source of truth. Never substitute neighboring text, complete cut-off sentences, summarize, or invent content.
Keep every source block in the same order. Preserve the source block IDs as HTML data-source-block attributes when practical.
Output Bangla as primarily LTR, but preserve legitimate embedded Arabic/Urdu quotations in local spans with dir="rtl"; never globally rewrite direction.
Convert narrative numbers to Bangla numerals. Preserve digits in citations, footnote markers, page/volume/issue references, and formal dates.
Translate headers, footers, metadata, lists, tables, and footnotes. For each visual element, output a placeholder with a stable data-diagram-id such as page-${page.id}-visual-1.
Use inline CSS and preserve semantic colors: indigo headings, slate body text, rose highlights, muted footnotes, and distinct quote/citation styling.
Document glossary (hints only; contextual correctness wins): ${context.glossaryContext || "none"}.
Nearby heading context (context only, never a completion): ${context.continuationContext}.
STRUCTURED SOURCE BLOCKS:
${context.blockContext}
RAW SOURCE TEXT:
${page.content.rawText}
Return raw HTML only.`;
  };

  const buildVerifyPrompt = (page, draft) => `${buildTranslationPrompt(page)}

You are now auditing the existing Bangla draft below, not translating from scratch. Fix only real omissions, substitutions, hallucinated completions, source-language leftovers, missing visual placeholders, and footnote mismatches. Preserve correct wording and styling. Return exactly NO_CORRECTIONS_NEEDED or the complete corrected raw HTML.
EXISTING BANGLA DRAFT:
${draft}`;

  const translatePage = async (page, useFlash) => {
    const image = await extractPageImageBase64(page.id);
    const parts = [{ text: buildTranslationPrompt(page) }];
    if (image) parts.push({ inlineData: { mimeType: "image/jpeg", data: image } });
    const data = await callGemini(parts, useFlash);
    const html = ensureVisualIdentity(cleanModelHtml(data.candidates?.[0]?.content?.parts?.[0]?.text || ""), page.id);
    const validation = validateTranslation(html, page.content.rawText);
    return { html, validation };
  };

  const verifyPage = async (page, draft) => {
    const image = await extractPageImageBase64(page.id);
    const parts = [{ text: buildVerifyPrompt(page, draft) }];
    if (image) parts.push({ inlineData: { mimeType: "image/jpeg", data: image } });
    const data = await callGemini(parts, false);
    const raw = cleanModelHtml(data.candidates?.[0]?.content?.parts?.[0]?.text || "");
    const html = /^NO_CORRECTIONS_NEEDED$/i.test(raw) ? draft : ensureVisualIdentity(raw, page.id);
    return { html, validation: validateTranslation(html, page.content.rawText) };
  };

  const beginJob = (pageId) => {
    const token = `${pageId}-${Date.now()}-${Math.random()}`;
    jobsRef.current.set(pageId, token);
    return token;
  };
  const isCurrentJob = (pageId, token) => jobsRef.current.get(pageId) === token;
  const markTranslation = (pageId, html, validation, engine, verificationStatus = "not-run") => {
    // GROUP 3: merge small document-level terminology memory without making
    // the glossary authoritative over a later page's correct context.
    const sourcePage = parsedSections.find((page) => page.id === pageId);
    const additions = sourcePage ? glossaryFromTranslation(sourcePage, html) : [];
    if (additions.length) {
      setGlossary((previous) => {
        const merged = [...previous, ...additions];
        return merged.filter((term, index, all) =>
          all.findIndex((candidate) => candidate.source === term.source) === index
        ).slice(-80);
      });
    }
    setPage(pageId, (page) => ({
      ...page,
      banglaHtml: html,
      originalBanglaHtml: page.originalBanglaHtml || html,
      translationStatus: "done",
      engineProduced: engine,
      verificationStatus,
      warnings: [...new Set([...(page.warnings || []), ...(validation.warnings || [])])],
      glossaryTerms: glossaryFromTranslation(page, html)
    }));
    if (validation.warnings?.length) setSuccessMsg("Translation saved with non-blocking review warnings.");
  };

  // GROUP 1: Retry always uses the default engine and the page's own ID/data.
  const retryPage = async (pageId) => {
    const page = parsedSections.find((item) => item.id === pageId);
    if (!page) return;
    const token = beginJob(pageId);
    setErrorMsg(""); setSuccessMsg("");
    setPage(pageId, (current) => ({ ...current, translationStatus: "loading" }));
    try {
      const draft = await translatePage(page, false);
      if (!isCurrentJob(pageId, token)) return;
      const checked = await verifyPage(page, draft.html);
      if (!isCurrentJob(pageId, token)) return;
      markTranslation(pageId, checked.html, { warnings: [...draft.validation.warnings, ...checked.validation.warnings] }, "default", "verified");
    } catch (error) {
      if (!isCurrentJob(pageId, token)) return;
      setPage(pageId, (current) => ({ ...current, translationStatus: "error", warnings: [...new Set([...(current.warnings || []), error.message])] }));
      setErrorMsg(error.message === "NO_API_KEY" ? "Add a Gemini API key in Settings." : "Translation failed. Use Retry after fixing the issue.");
    }
  };

  const startSequentialAnalysis = async () => {
    if (isTranslatingAll) return;
    setIsTranslatingAll(true); setProgress(0); setErrorMsg("");
    const pageIds = parsedSections.map((page) => page.id);
    for (let index = 0; index < pageIds.length; index += 1) {
      const pageId = pageIds[index];
      const page = parsedSections.find((item) => item.id === pageId);
      if (!page || page.translationStatus === "done") {
        setProgress(Math.round(((index + 1) / pageIds.length) * 100));
        continue;
      }
      await retryPage(pageId);
      setProgress(Math.round(((index + 1) / pageIds.length) * 100));
    }
    setIsTranslatingAll(false);
    setSuccessMsg("Batch translation and verification complete.");
  };

  const handleVerifyPage = async (pageId) => {
    const page = parsedSections.find((item) => item.id === pageId);
    if (!page?.banglaHtml) return;
    const token = beginJob(pageId);
    setPage(pageId, (current) => ({ ...current, translationStatus: "verifying" }));
    try {
      const result = await verifyPage(page, page.banglaHtml);
      if (!isCurrentJob(pageId, token)) return;
      markTranslation(pageId, result.html, result.validation, "verify-corrected", "verified");
      setSuccessMsg(`Page ${pageId} verified.`);
    } catch {
      if (isCurrentJob(pageId, token)) {
        setPage(pageId, (current) => ({ ...current, translationStatus: "done", verificationStatus: "failed" }));
        setErrorMsg("Verification failed; the existing translation was preserved.");
      }
    }
  };

  const retranslatePageFlash = async (pageId) => {
    const page = parsedSections.find((item) => item.id === pageId);
    if (!page) return;
    const token = beginJob(pageId);
    setPage(pageId, (current) => ({ ...current, translationStatus: "loading" }));
    try {
      const result = await translatePage(page, true);
      if (!isCurrentJob(pageId, token)) return;
      markTranslation(pageId, result.html, result.validation, "flash", "not-run");
      setSuccessMsg(`Page ${pageId} re-translated with the Flash cascade.`);
    } catch (error) {
      if (isCurrentJob(pageId, token)) {
        setPage(pageId, (current) => ({ ...current, translationStatus: "error" }));
        setErrorMsg(error.message === "ALL_KEYS_EXHAUSTED" ? "All Gemini keys/models are exhausted or invalid." : "Flash translation failed.");
      }
    }
  };

  const handleResetTranslation = (pageId) => {
    setPage(pageId, (page) => page.originalBanglaHtml ? ({
      ...page, banglaHtml: page.originalBanglaHtml, translationStatus: "done",
      engineProduced: "original snapshot", verificationStatus: "not-run"
    }) : page);
    setSuccessMsg(`Page ${pageId} reset to its first successful translation.`);
  };

  const handleEditableBlur = (pageId, event) => {
    const html = ensureVisualIdentity(event.currentTarget.innerHTML, pageId);
    setPage(pageId, (page) => ({ ...page, banglaHtml: html }));
  };

  const handleSaveSettings = () => {
    const keys = apiKeys.map((key) => key.trim());
    if (!keys[0]) return setErrorMsg("Please enter at least the first Gemini API key.");
    sessionStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
    if (rememberApiKey) localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
    else localStorage.removeItem(API_KEYS_STORAGE_KEY);
    setApiKeys(keys); rotationRef.current = makeRotation(); flashRotationRef.current = makeRotation();
    setShowSettings(false); setSuccessMsg("Settings saved. Keys remain browser-side only.");
  };

  // GROUP 4: replacements are computed on a detached clone, then persisted
  // through the page's stable data-diagram-id.
  useEffect(() => {
    const onChange = (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement) || !input.matches(".diagram-upload-input,.diagram-replace-input")) return;
      const file = input.files?.[0];
      const pageId = Number(input.closest("[data-editor-page]")?.getAttribute("data-editor-page"));
      const diagramId = input.closest("[data-diagram-id]")?.getAttribute("data-diagram-id");
      if (!file || !pageId || !diagramId) return;
      const reader = new FileReader();
      reader.onload = () => {
        const page = parsedSections.find((item) => item.id === pageId);
        if (!page) return;
        const detached = document.createElement("div");
        detached.innerHTML = page.banglaHtml;
        const placeholder = detached.querySelector(`[data-diagram-id="${CSS.escape(diagramId)}"]`);
        if (!placeholder) return;
        placeholder.outerHTML = `<div class="diagram-placeholder uploaded-visual" data-diagram-id="${diagramId}" style="text-align:center;margin:16px 0;"><img src="${reader.result}" alt="Uploaded diagram" style="max-width:100%;max-height:400px;border-radius:8px;display:block;margin:0 auto;box-shadow:0 4px 6px -1px rgb(0 0 0 / .1);" /><div class="visual-controls"><label class="visual-control">Replace Image<input type="file" accept="image/*" class="diagram-replace-input" /></label><button type="button" data-action="remove-image" data-diagram-id="${diagramId}">Remove Image</button></div></div>`;
        setPage(pageId, (current) => ({ ...current, banglaHtml: safeSanitize(detached.innerHTML) }));
        setSuccessMsg("Image replacement saved.");
      };
      reader.readAsDataURL(file);
      input.value = "";
    };
    const onClick = (event) => {
      const button = event.target.closest("[data-action='remove-image']");
      if (!button) return;
      const editor = button.closest("[data-editor-page]");
      const pageId = Number(editor?.getAttribute("data-editor-page"));
      const diagramId = button.getAttribute("data-diagram-id");
      const page = parsedSections.find((item) => item.id === pageId);
      if (!page || !diagramId) return;
      const detached = document.createElement("div");
      detached.innerHTML = page.banglaHtml;
      const placeholder = detached.querySelector(`[data-diagram-id="${CSS.escape(diagramId)}"]`);
      if (!placeholder) return;
      placeholder.outerHTML = `<div class="diagram-placeholder" data-diagram-id="${diagramId}" style="border:2px dashed #CBD5E1;padding:20px;text-align:center;border-radius:8px;margin:16px 0;"><p style="font-size:14px;color:#64748B;margin-bottom:8px;">Image detected. Upload a replacement.</p><label class="visual-control">Upload Image<input type="file" accept="image/*" class="diagram-upload-input" /></label></div>`;
      setPage(pageId, (current) => ({ ...current, banglaHtml: safeSanitize(detached.innerHTML) }));
      setSuccessMsg("Image removed; placeholder restored.");
    };
    document.addEventListener("change", onChange);
    document.addEventListener("click", onClick);
    return () => { document.removeEventListener("change", onChange); document.removeEventListener("click", onClick); };
  }, [parsedSections, setPage]);

  const handleCopy = async (html, id) => {
    try {
      await navigator.clipboard.writeText(textOnly(html));
      setCopiedId(id); window.setTimeout(() => setCopiedId(""), 1500);
    } catch { setErrorMsg("Could not copy text."); }
  };

  const exportHtml = async () => {
    if (isExporting) return;
    const translated = parsedSections.filter((page) => page.translationStatus === "done" && page.banglaHtml);
    if (!translated.length) return setErrorMsg("No translated pages found.");
    const ids = translated.map((page) => page.id);
    const duplicate = ids.some((id, index) => ids.indexOf(id) !== index);
    const malformed = translated.filter((page) => !safeSanitize(page.banglaHtml).trim());
    if (duplicate || malformed.length) return setErrorMsg("Export warning: duplicate, empty, or malformed page sections need review.");
    setIsExporting(true);
    const sections = translated.map((page) => `
      <section class="translated-page" id="page-${page.id}" data-page-id="${page.id}" style="--page-ratio:${page.geometry.aspectRatio};">
        <header class="page-label">Page ${page.id}</header>
        <div class="page-content" dir="ltr">${safeSanitize(page.banglaHtml)}</div>
      </section>`).join("\n");
    const html = `<!doctype html><html lang="bn"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Translated Bangla - ${fileData?.name || "Document"}</title>
      <link rel="stylesheet" href="https://fonts.maateen.me/kalpurush/font.css"><style>
      *{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font-family:Kalpurush, sans-serif;padding:32px 16px}
      .translated-page{width:min(850px,100%);min-height:calc(850px / var(--page-ratio));margin:0 auto 36px;background:#fff;border:1px solid #e2e8f0;border-radius:12px;box-shadow:0 10px 30px #0f172a12;padding:38px 44px;position:relative;break-after:page;page-break-after:always}
      .translated-page:last-child{break-after:auto;page-break-after:auto}.page-label{color:#64748b;border-bottom:1px dashed #cbd5e1;padding-bottom:10px;margin-bottom:24px;font-family:Arial,sans-serif;font-size:12px;font-weight:700}
      .page-content{font-size:18px;line-height:1.8}.page-content .footnotes{color:#64748b;font-size:14px;border-top:1px solid #e2e8f0;margin-top:24px;padding-top:12px}
      .page-content table{width:100%;border-collapse:collapse}.page-content td,.page-content th{border:1px solid #cbd5e1;padding:6px}.page-content [dir="rtl"]{font-family:"Noto Naskh Arabic",serif}
      @media print{body{background:#fff;padding:0}.translated-page{width:100%;min-height:auto;border:0;box-shadow:none;border-radius:0;margin:0;padding:20px 0}.page-label{display:none}}
      </style></head><body>${sections}</body></html>`;
    const url = URL.createObjectURL(new Blob([html], { type: "text/html;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `Translated_Bangla_${(fileData?.name || "Document").replace(/\.pdf$/i, "")}.html`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
    setIsExporting(false); setSuccessMsg("Hardened HTML export downloaded.");
  };

  const activePage = parsedSections.find((page) => page.id === activeSectionId);
  const activeWarnings = useMemo(() => activePage?.warnings || [], [activePage]);
  const statusLabel = (status) => ({ idle: "Not processed", loading: "Translating…", verifying: "Verifying…", error: "Failed", done: "Translated" }[status] || status);

  const renderPageActions = (page) => (
    <div className="page-actions">
      {page.translationStatus === "error" && <button className="small-button retry" onClick={() => retryPage(page.id)}><RotateCcw size={13} /> Retry</button>}
      {page.translationStatus !== "idle" && <button className="icon-button" disabled={["loading", "verifying"].includes(page.translationStatus) || !page.banglaHtml} title="Verify with default Flash-Lite engine" onClick={() => handleVerifyPage(page.id)}><ShieldCheck size={14} /></button>}
      {page.translationStatus !== "idle" && <button className="icon-button" disabled={["loading", "verifying"].includes(page.translationStatus)} title="Flash cascade: 3.6 → 3.5 → 3.8 → 3.7 → 3-preview" onClick={() => retranslatePageFlash(page.id)}><Zap size={14} /></button>}
      {page.translationStatus !== "idle" && <button className="icon-button" disabled={["loading", "verifying"].includes(page.translationStatus) || !page.originalBanglaHtml} title="Reset to first successful translation" onClick={() => handleResetTranslation(page.id)}><RotateCcw size={14} /></button>}
    </div>
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><BookOpen size={20} /></div><div><h1>Ar/En/Ur to Bangla Translator</h1><span><Globe size={11} /> Multilingual mirror layout studio</span></div></div>
        <div className="top-actions">
          <div className="language-switch">
            {["auto", "ar_ur", "en"].map((mode) => <button key={mode} className={sourceLangMode === mode ? "active" : ""} onClick={() => setSourceLangMode(mode)}>{mode === "auto" ? "Auto" : mode === "ar_ur" ? "العربية / اردو" : "English"}</button>)}
          </div>
          {parsedSections.length > 0 && <><button className="primary-button" onClick={startSequentialAnalysis} disabled={isTranslatingAll}>{isTranslatingAll ? <><Loader2 className="spin" size={15} /> {progress}%</> : <><Sparkles size={15} /> Translate Entire Document</>}</button><button className="dark-button" onClick={exportHtml} disabled={isExporting}>{isExporting ? <Loader2 className="spin" size={15} /> : <Download size={15} />} Export HTML</button></>}
          <button className="plain-icon" onClick={() => setShowSettings(true)} title="Settings"><Settings size={18} /></button>
          <button className="plain-icon" onClick={() => setShowHelp(true)} title="Help"><HelpCircle size={18} /></button>
        </div>
      </header>

      {showSettings && <div className="modal-backdrop"><div className="modal"><div className="modal-title"><h2><Settings size={18} /> Application Settings</h2><button onClick={() => setShowSettings(false)}><X size={18} /></button></div><p className="muted">Keys are sent directly from this browser to Gemini. No server or Replit Secret is used.</p><div className="key-list">{apiKeys.map((key, index) => <label key={index}>API Key {index + 1}{index === 0 ? " (required)" : " (optional)"}<input type="password" value={key} onChange={(event) => setApiKeys((prev) => prev.map((item, i) => i === index ? event.target.value : item))} placeholder="Gemini API key" /></label>)}</div><label className="check-row"><input type="checkbox" checked={rememberApiKey} onChange={(event) => setRememberApiKey(event.target.checked)} /> Remember keys on this device (localStorage)</label><label className="field-label">Source language<select value={sourceLangMode} onChange={(event) => setSourceLangMode(event.target.value)}><option value="auto">Auto detect</option><option value="ar_ur">Arabic / Urdu</option><option value="en">English</option></select></label><div className="modal-footer"><button className="primary-button" onClick={handleSaveSettings}>Save & Close</button></div></div></div>}
      {showHelp && <div className="modal-backdrop"><div className="modal help"><div className="modal-title"><h2><Sparkles size={18} /> Workflow guide</h2><button onClick={() => setShowHelp(false)}><X size={18} /></button></div><p>Upload a PDF, translate pages one at a time, then review with Verify, Flash, Compare, or Overlay. Content is editable and export is read-only.</p><ul><li><b>Retry</b> uses the default Flash-Lite two-pass flow.</li><li><b>Flash</b> uses its separate five-model cascade.</li><li><b>Reset</b> always returns to the first successful translation.</li></ul><button className="dark-button" onClick={() => setShowHelp(false)}>Got it</button></div></div>}

      <main className="main-area">
        {isParsing ? <div className="empty-state"><Loader2 className="spin large" /><h2>Reading document…</h2><p>Extracting text geometry and structure.</p><div className="progress"><span style={{ width: `${parseProgress}%` }} /></div><b>{parseProgress}%</b></div>
          : !fileData ? <div className="empty-state"><Languages size={40} className="indigo" /><h2>Preserve layouts. Translate to Bangla.</h2><p>Upload an Arabic, Urdu, or English PDF to reconstruct its pages with structured metadata and a client-side Gemini workflow.</p><label className="upload-card"><Upload size={28} /><strong>Browse PDF file</strong><span>PDF only · maximum 100MB</span><input ref={fileInputRef} type="file" accept=".pdf,application/pdf" onChange={handleFileUpload} /></label></div>
          : <div className="workspace">
            <aside className="sidebar"><div className="file-heading"><div className="file-icon"><FileText size={16} /></div><div><b title={fileData.name}>{fileData.name}</b><small>{parsedSections.length} pages · {fileData.size} MB</small></div><button className="plain-icon" onClick={() => { localStorage.removeItem(SESSION_STORAGE_KEY); window.location.reload(); }} title="Close document"><X size={15} /></button></div><div className="sidebar-label">Document navigation</div><div className="page-list">{parsedSections.map((page) => <div key={page.id} className={`page-row ${page.id === activeSectionId ? "selected" : ""}`}><button className="page-select" onClick={() => setActiveSectionId(page.id)}><span className={`page-number ${page.translationStatus}`}>{page.id}</span><span><b>{page.meta.partName}</b><small>{statusLabel(page.translationStatus)}</small></span><ChevronRight size={14} /></button>{renderPageActions(page)}</div>)}</div></aside>
            <section className="content-area"><div className="workspace-bar"><span>Workspace:</span><b>Page {activeSectionId}</b><div className="view-switch"><button className={reviewMode === "target" ? "active" : ""} onClick={() => setReviewMode("target")}><Eye size={13} /> Target</button><button className={reviewMode === "compare" ? "active" : ""} onClick={() => setReviewMode("compare")}><Languages size={13} /> Compare</button><button className={reviewMode === "overlay" ? "active" : ""} onClick={() => setReviewMode("overlay")}><ImagePlus size={13} /> Overlay</button></div></div>
              <div className="review-stage">
                {!activePage ? null : activePage.translationStatus === "idle" ? <div className="panel-message"><Globe size={26} /><h3>Not yet translated</h3><p>Use Translate Entire Document to generate the Bangla page.</p></div>
                  : activePage.translationStatus === "loading" || activePage.translationStatus === "verifying" ? <div className="panel-message"><Loader2 className="spin" size={30} /><h3>{activePage.translationStatus === "verifying" ? "Verifying and correcting…" : "Translating to Bangla…"}</h3><p>Only this page is being updated.</p></div>
                  : activePage.translationStatus === "error" ? <div className="panel-message"><AlertCircle size={30} className="red" /><h3>Translation failed</h3><p>Nothing was replaced. Retry this page through the default engine.</p><button className="primary-button" onClick={() => retryPage(activePage.id)}><RotateCcw size={14} /> Retry page</button></div>
                  : <div className={`review-grid ${reviewMode}`}><div className="paper original-paper" style={{ display: reviewMode === "target" ? "none" : undefined }}><div className="paper-label">Original PDF page</div>{activeSourceImage ? <img src={activeSourceImage} alt={`Original page ${activePage.id}`} /> : <p className="muted">Original PDF image is unavailable after a refresh; translation remains available.</p>}</div><div className="paper target-paper" data-editor-page={activePage.id} style={{ position: "relative" }}><div className="paper-label"><span>Bangla target layout</span><span className="engine-chip">{activePage.engineProduced || "default"} · {activePage.verificationStatus}</span></div>{reviewMode === "overlay" && activeSourceImage && <img className="overlay-image" style={{ opacity: overlayOpacity }} src={activeSourceImage} alt="" />}<div className="editable-page" contentEditable suppressContentEditableWarning onBlur={(event) => handleEditableBlur(activePage.id, event)} dangerouslySetInnerHTML={{ __html: safeSanitize(activePage.banglaHtml) }} />{reviewMode === "overlay" && <label className="opacity-control">Overlay opacity <input type="range" min="0" max="1" step="0.05" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} /></label>}</div></div>}
              </div>
              {activePage && activePage.translationStatus === "done" && <div className="diagnostics"><div className="diagnostics-title"><span><ShieldCheck size={14} /> Page diagnostics</span><button className="plain-icon" onClick={() => handleCopy(activePage.banglaHtml, `copy-${activePage.id}`)}>{copiedId === `copy-${activePage.id}` ? <Check size={14} /> : <Copy size={14} />}</button></div><div className="diagnostic-grid"><span>Source blocks <b>{activePage.blocks.length}</b></span><span>Visual assets <b>{activePage.visuals.length}</b></span><span>Footnotes <b>{activePage.blocks.filter((block) => block.type === "footnote").length}</b></span><span>Engine <b>{activePage.engineProduced}</b></span></div>{activeWarnings.length > 0 && <div className="warning-list">{activeWarnings.map((warning, index) => <span key={index}><AlertCircle size={12} /> {warning}</span>)}</div>}</div>}
            </section>
          </div>}
      </main>
      {errorMsg && <div className="toast error"><AlertCircle size={17} /><span>{errorMsg}</span><button onClick={() => setErrorMsg("")}><X size={14} /></button></div>}
      {successMsg && <div className="toast success"><CheckCircle2 size={17} /><span>{successMsg}</span><button onClick={() => setSuccessMsg("")}><X size={14} /></button></div>}
    </div>
  );
};

export default App;
import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Loader2,
  CheckCircle2,
  X,
  FileText,
  AlertCircle,
  ChevronRight,
  BookOpen,
  Settings,
  Globe,
  FileDown,
  Sparkles,
  HelpCircle,
  Copy,
  Check,
  ShieldCheck
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Exponential backoff fetch implementation with timeout and abort handling.
// `acquireSlot`, when provided, is awaited before EVERY attempt (including
// retries) so no HTTP call - first try or retry - can ever bypass the shared
// rate-limit pacing.
// ---------------------------------------------------------------------------
const fetchWithRetry = async (url, options, timeoutMs = 60000, acquireSlot = null) => {
  const delays = [1000, 2000, 4000, 8000, 16000];
  let lastError;

  for (let i = 0; i <= delays.length; i++) {
    if (acquireSlot) {
      await acquireSlot();
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text().catch(() => "");
        const statusError = new Error(`API Error ${response.status}: ${errorText || response.statusText}`);
        statusError.status = response.status;
        throw statusError;
      }
      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      lastError = error;

      // 429 (quota) and 400/401/403 (bad key) are handled by the caller's key
      // rotation logic, so surface them immediately instead of backing off.
      if (error.status === 429 || error.status === 400 || error.status === 401 || error.status === 403) {
        throw error;
      }

      if (error.name === 'AbortError') {
        console.warn(`Request timed out after ${timeoutMs}ms, retrying... (Attempt ${i + 1})`);
      } else {
        console.warn(`Fetch failed: ${error.message}. Retrying... (Attempt ${i + 1})`);
      }

      if (i < delays.length) {
        await new Promise(resolve => setTimeout(resolve, delays[i]));
      }
    }
  }
  throw lastError;
};

// ---------------------------------------------------------------------------
// SINGLE MODEL ENGINE - Ministral 3 14B only. No cascade, no free-tier daily
// request caps (this is a paid, per-token API, not a rate-limited free tier).
// ---------------------------------------------------------------------------
const MISTRAL_MODEL = 'ministral-14b-2512';
// IMPORTANT: browsers cannot call api.mistral.ai directly (Mistral does not
// allow cross-origin browser requests - CORS). Point this at your own proxy
// server instead (see mistral-proxy-server.js / server.js). Replace the URL
// below with YOUR CodeSandbox proxy's public URL + "/api/translate".
const MISTRAL_API_URL = 'https://mistral-proxy-wk5t.onrender.com/api/translate';

// Exactly 5 key slots, rotated in strict round-robin order: 1 -> 2 -> 3 -> 4
// -> 5 -> back to 1 -> ... A key is rotated away from the moment it returns a
// quota/limit error (e.g. its $10 cap is hit) or an auth error (invalid key).
const TOTAL_API_KEY_SLOTS = 5;

// Mistral API keys are 25-40 characters long.
const MIN_MISTRAL_KEY_LENGTH = 25;
const MAX_MISTRAL_KEY_LENGTH = 40;

// Minimum 2 second gap between EVERY outgoing request (first attempt or retry).
const REQUEST_INTERVAL_MS = 2000;

const LAST_REQUEST_TIME_STORAGE_KEY = 'ut_translator_last_request_time_v1';
const API_KEYS_STORAGE_KEY = 'ut_translator_api_keys_v1';
const SESSION_STORAGE_KEY = 'ut_translator_session_v1';

// ---------------------------------------------------------------------------
// Prompt builders - identical guard-rail logic as before (content-substitution
// guard, footnote handling, layout fidelity, visual-scan for diagrams), just
// parameterized by target language so one engine serves both Bangla and
// English output.
// ---------------------------------------------------------------------------
const TARGET_LABELS = {
  bn: 'Bangla',
  en: 'English'
};

const buildTranslatePrompt = (targetCode, langInstruction, pageText) => {
  const targetName = TARGET_LABELS[targetCode];
  const numeralLine = targetCode === 'bn'
    ? ' - convert numerals to Bangla numerals too'
    : '';
  const transliterationLine = targetCode === 'bn'
    ? 'Preserve Quranic verses / Arabic religious terms with standard Bangla transliteration or brief explanation where appropriate.'
    : 'Preserve Quranic verses / Arabic religious terms with standard English transliteration or brief explanation where appropriate.';

  return `
      READ THIS SOURCE PAGE (ARABIC / URDU / ENGLISH) AND PRODUCE ITS 100% MIRRORED ${targetName.toUpperCase()} TRANSLATION AS STYLED HTML, IN ONE SINGLE PASS.

      ROLE: You are an expert scholarly translator. Your single most important job, above everything else in this prompt, is: translate exactly the sentence that is actually printed, in exactly the place it is printed, into faithful ${targetName} - never a different (even if related) sentence from elsewhere on the page.
      ${langInstruction}

      ============================================================
      RULE #1 - THE MOST IMPORTANT RULE - NO CONTENT SUBSTITUTION (READ THIS FIRST, APPLY IT ABOVE ALL ELSE)
      ============================================================
      This document contains dense, technical, citation-heavy academic/religious content (hadith science terminology, multi-narrator citation chains, scholarly argumentation). On pages like this, there is a known failure mode where a translator (human or AI) under cognitive load accidentally substitutes a DIFFERENT nearby sentence - often a quotation, footnote, or an adjacent paragraph discussing a related concept - in place of the sentence that was actually supposed to be translated at that position. This is a much more serious error than a wrong word choice: it silently replaces the meaning of an entire paragraph.
      YOU MUST ACTIVELY GUARD AGAINST THIS. For every paragraph, before finalizing its ${targetName} translation, re-confirm: "Is this translation actually derived from THIS SPECIFIC paragraph's own words, or did I drift into translating a different quotation/footnote/paragraph that happens to be nearby and discusses a similar topic?" If you notice any drift, discard it and re-translate strictly from the actual source sentence at that position, even if the correct translation reads as less polished, less complete, or less like a "clean quotable passage" than the substituted version would have.
      When a paragraph contains an embedded quotation (e.g. «...» or "..."), translate the quotation as part of that paragraph, in that exact position - do not let a quotation's content bleed into a neighboring paragraph, and do not let a neighboring paragraph's content bleed into the quotation.
      This rule OVERRIDES any instinct to produce smoother-sounding or more "complete" ${targetName} text. A literal, faithful translation of the correct sentence is always better than a fluent translation of the wrong sentence.

      ============================================================
      RULE #2 - NEVER OUTPUT THE ORIGINAL-LANGUAGE TEXT
      ============================================================
      The original Arabic/Urdu/English source text must NEVER appear anywhere in your output - only the ${targetName} translation is ever shown, even for the densest citation-chain passages. Do not preserve the original "for precision" alongside the translation.

      RAW TEXT CONTENT (extracted from the page's text layer - may be empty/garbled if this is a scanned/image-only page):
      """
      ${pageText}
      """

      --- IMAGE OCR & BLANK PAGE HANDLING ---
      If "RAW TEXT CONTENT" is empty, garbage, or incomplete (e.g. a scanned image-only PDF), rely ENTIRELY on the provided page image to perform OCR and read the full source text accurately before translating.
      If the image and text BOTH contain no readable content (a genuinely blank page), ignore all layout rules and output EXACTLY:
      <p style="text-align: center; color: #94A3B8; font-style: italic; font-size: 16px; margin-top: 40px;">No Text Found</p>

      --- LAYOUT FIDELITY ---
      Preserve the EXACT paragraph breaks, lists, tables, and physical structure of the original page - do not re-order, combine, or split paragraphs, and do not summarize, omit, or add any text or meaning that isn't in the source. Only the LANGUAGE changes, to ${targetName}; the structure stays a 1:1 mirror.

      --- TRANSLATE EVERY VISIBLE TEXT ELEMENT - NO EXCEPTIONS ---
      Every visible piece of text on the page must be translated, including running headers/footers, journal/publication names, issue numbers, dates, and page numbers (small metadata strips at the top/bottom included)${numeralLine}. Nothing is left in the original language anywhere on the page.

      --- FOOTNOTE HANDLING ---
      Read the body text first, keeping every footnote reference marker (superscript number/symbol) exactly in place - the marker is a structural element and must survive into the translation, but the footnote's MEANING must never be used to complete, clarify, or supplement the body (see RULE #1 above: proximity to a footnote is exactly the kind of situation where content substitution/bleeding happens - stay anchored to the body's own actual sentence). Read and translate the footnote block separately. Cross-check that every marker in the body has a matching footnote entry and vice versa. After drafting, compare body vs. footnote translations sentence-by-sentence: if any exact sentence appears in both, that's an error - remove the duplicate from the body and restore the body's own faithful (possibly cut-off) translation.

      --- NO INVENTED COMPLETIONS ---
      If a sentence, heading, or list item appears cut off at the bottom of the page, translate it AS CUT OFF - do not invent a completion from your own knowledge (even of a well-known hadith/verse), and do not add trailing punctuation that wasn't in the source.

      --- DEDICATED VISUAL SCAN STEP ---
      Separately from reading the text, visually scan the ENTIRE page image for any non-text visual element: photographs, diagrams, charts, graphs, maps, tables-as-images, illustrations, icons, stamps, seals, logos, letterheads, infographics, screenshots, decorative borders/dividers. Logos/seals/stamps are the most commonly missed - they are often small, subtle, monochrome, or near a title, and can be mistaken for decorative heading design; check for them specifically rather than assuming a mark near a title is just styling.
      For EACH element found, output this EXACT HTML in its correct spatial place:
      <div class="diagram-placeholder" style="border: 2px dashed #CBD5E1; padding: 20px; text-align: center; border-radius: 8px; margin: 16px 0;"><p style="font-size: 14px; color: #64748B; margin-bottom: 8px;">Image Detected. Click to upload replacement.</p><input type="file" accept="image/*" class="diagram-upload-input" /></div>

      --- BEAUTIFICATION & STYLING (INLINE CSS ONLY, OUTPUT IS ${targetName.toUpperCase()}) ---
      Output text is always ${targetName}, always LTR (direction: ltr;), regardless of source direction.
      1. Heading 1: <p style="text-align: center; color: #4338CA; font-size: 24px; font-weight: bold; margin-bottom: 20px;">...</p>
      2. Heading 2: <p style="color: #0F172A; font-size: 20px; font-weight: bold; margin-bottom: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">...</p>
      3. Body text: <p style="text-align: justify; color: #334155; font-size: 18px; line-height: 1.8; margin-bottom: 16px;">...</p>
      4. Highlights: <span style="color: #BE123C; font-weight: bold;">...</span> for emphasis/Quranic verses/key terms.
      5. Footnotes: <div class="footnotes" style="margin-top: 24px; border-top: 1px solid #E2E8F0; padding-top: 12px;"><p style="color: #64748B; font-size: 14px;">...</p></div>
      6. ${transliterationLine}

      DO NOT use markdown code blocks (\`\`\`html). Output raw styled HTML directly.
    `;
};

const buildVerifyPrompt = (targetCode, pageText, translatedHtmlDraft) => {
  const targetName = TARGET_LABELS[targetCode];
  return `
      YOU ARE A PROOFREADER/AUDITOR, NOT A TRANSLATOR. Your job is to CHECK an existing ${targetName} translation against its original source page, and fix ONLY real errors you find - not to re-translate from scratch or rewrite style choices you simply would have phrased differently.

      ORIGINAL SOURCE PAGE TEXT (Arabic / Urdu / English - also see attached page image if provided):
      """
      ${pageText}
      """

      EXISTING ${targetName.toUpperCase()} TRANSLATION HTML TO AUDIT:
      """
      ${translatedHtmlDraft}
      """

      Check specifically for these failure modes, one by one:
      1. CONTENT SUBSTITUTION: does each paragraph's ${targetName} actually correspond to THAT SAME paragraph's source text, or did content drift in from a nearby quotation, footnote, or adjacent paragraph discussing a related topic?
      2. HALLUCINATION / INVENTED COMPLETIONS: did the translation add words, sentences, or punctuation not present in the source - especially completing a sentence that was actually cut off at the bottom of the page?
      3. FOOTNOTE MARKERS: does every footnote reference marker (superscript number/symbol) in the body have a matching footnote entry, and vice versa? Did any footnote's meaning bleed into the body text (or an identical sentence appear in both)?
      4. ORIGINAL-LANGUAGE LEFTOVERS: does any Arabic/Urdu/English text remain anywhere in the output that should have been translated to ${targetName}?
      5. VISUAL ELEMENT PLACEHOLDERS: compare the page image (if provided) against the HTML - is any photograph, diagram, chart, graph, map, table-as-image, illustration, icon, stamp, seal, or logo visible on the page missing its placeholder <div class="diagram-placeholder">...</div> block?
      6. HEADERS/FOOTERS/PAGE NUMBERS: is every peripheral text element (running header/footer, journal name, issue number, date, page number) fully translated to ${targetName}${targetCode === 'bn' ? ', including numerals' : ''}?

      OUTPUT RULES:
      - If you find NO issues after checking all six points above, respond with EXACTLY this sentinel text and nothing else: NO_CORRECTIONS_NEEDED
      - If you find ANY issue, respond with the FULL corrected HTML (same format/styling conventions as the input - styled <p>/<div>/<span> tags, ${targetName} text, LTR direction), with ONLY the specific errors fixed. Do not rewrite or rephrase parts that were already correct - preserve everything that wasn't actually wrong.
      - Do not use markdown code blocks. Output either the sentinel text alone, or raw HTML alone - never both, never any other commentary.
    `;
};

const App = () => {
  // File / section state
  const [fileData, setFileData] = useState(null);
  const [parsedSections, setParsedSections] = useState([]);
  const [activeSectionId, setActiveSectionId] = useState(null);
  const [isParsing, setIsParsing] = useState(false);
  const [parseProgress, setParseProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);

  // Translation workflow state
  const [sourceLangMode, setSourceLangMode] = useState('auto'); // 'auto', 'ar_ur', 'en'
  const [targetLang, setTargetLang] = useState('bn'); // 'bn' (Bangla, all 3 sources) or 'en' (English, Ar/Ur only)
  const [isTranslatingAll, setIsTranslatingAll] = useState(false);
  const [progress, setProgress] = useState(0);

  // Settings & API keys (Mistral, 5 slots)
  const [apiKeys, setApiKeys] = useState(() => {
    try {
      const raw = sessionStorage.getItem(API_KEYS_STORAGE_KEY) || localStorage.getItem(API_KEYS_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length === TOTAL_API_KEY_SLOTS) return parsed;
      }
    } catch (e) { /* ignore */ }
    return Array(TOTAL_API_KEY_SLOTS).fill("");
  });
  const [rememberApiKey, setRememberApiKey] = useState(() => !!localStorage.getItem(API_KEYS_STORAGE_KEY));
  const [showSettings, setShowSettings] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const pdfDocRef = useRef(null);
  // keyIdx: which of the 5 keys is currently active. deadKeys: keys that
  // returned a hard auth error (invalid key) and are permanently skipped for
  // this session. Quota (429) errors just advance keyIdx round-robin without
  // marking the key dead, since it may work again once its cap resets.
  const rotationRef = useRef({ keyIdx: 0, deadKeys: new Set() });
  const throttleRef = useRef(Promise.resolve());
  const lastRequestTimeRef = useRef((() => {
    try {
      const saved = Number(localStorage.getItem(LAST_REQUEST_TIME_STORAGE_KEY));
      return Number.isFinite(saved) ? saved : 0;
    } catch (e) {
      return 0;
    }
  })());
  const hasRestoredSession = useRef(false);

  // Restore in-progress session
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SESSION_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && Array.isArray(parsed.parsedSections) && parsed.parsedSections.length > 0) {
          setFileData(parsed.fileData || null);
          setParsedSections(parsed.parsedSections);
          setActiveSectionId(parsed.activeSectionId ?? parsed.parsedSections[0].id);
          setSourceLangMode(parsed.sourceLangMode || 'auto');
          setTargetLang(parsed.targetLang || 'bn');
          setSuccessMsg("Restored your previous in-progress session.");
        }
      }
    } catch (e) {
      console.warn("Could not restore previous session:", e);
    } finally {
      hasRestoredSession.current = true;
    }
  }, []);

  // Persist progress after each change
  useEffect(() => {
    if (!hasRestoredSession.current) return;
    try {
      if (parsedSections.length > 0) {
        const payload = JSON.stringify({ fileData, parsedSections, activeSectionId, sourceLangMode, targetLang });
        localStorage.setItem(SESSION_STORAGE_KEY, payload);
      } else {
        localStorage.removeItem(SESSION_STORAGE_KEY);
      }
    } catch (e) {
      console.warn("Could not persist session (storage quota likely exceeded):", e);
    }
  }, [fileData, parsedSections, activeSectionId, sourceLangMode, targetLang]);

  // Enforce the valid source/target combinations: English target only makes
  // sense for Arabic/Urdu sources (English -> English is meaningless).
  useEffect(() => {
    if (targetLang === 'en' && sourceLangMode === 'en') {
      setSourceLangMode('auto');
    }
  }, [targetLang, sourceLangMode]);

  useEffect(() => {
    const twScript = document.createElement('script');
    twScript.src = 'https://cdn.tailwindcss.com';
    document.head.appendChild(twScript);

    const styleBlock = document.createElement('style');
    styleBlock.type = 'text/tailwindcss';
    styleBlock.innerHTML = `
        @import "tailwindcss";
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.maateen.me/kalpurush/font.css');
        @import url('https://fonts.googleapis.com/css2?family=Scheherazade+New:wght@400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

        @theme {
          --font-bangla: "Kalpurush", sans-serif;
          --font-arabic-urdu: "Scheherazade New", "Noto Naskh Arabic", serif;
          --font-sans: "Plus Jakarta Sans", sans-serif;
        }

        body {
          @apply bg-slate-50 text-slate-900 font-sans antialiased;
        }

        .bangla-font {
          font-family: "Kalpurush", sans-serif !important;
        }

        .english-font {
          font-family: "Times New Roman", "Plus Jakarta Sans", serif !important;
        }

        .arabic-urdu-font {
          font-family: "Scheherazade New", "Noto Naskh Arabic", serif !important;
        }

        .mirror-flow {
          @apply leading-relaxed tracking-normal bg-white p-6 md:p-8 rounded-xl border border-slate-100 transition-all;
        }

        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          @apply bg-transparent;
        }
        ::-webkit-scrollbar-thumb {
          @apply bg-slate-200 rounded-full hover:bg-slate-300;
        }
    `;
    document.head.appendChild(styleBlock);

    const pdfScript = document.createElement('script');
    pdfScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js';
    pdfScript.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';
    };
    document.head.appendChild(pdfScript);

    const purifyScript = document.createElement('script');
    purifyScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js';
    document.head.appendChild(purifyScript);

    return () => {
      [twScript, styleBlock, pdfScript, purifyScript].forEach(el => el?.remove());
    };
  }, []);

  const sanitizeHtml = useCallback((html) => {
    if (!html) return html;
    if (window.DOMPurify) {
      return window.DOMPurify.sanitize(html, {
        ADD_TAGS: ['input'],
        ADD_ATTR: ['class', 'accept', 'type']
      });
    }
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/ on[a-z]+="[^"]*"/gi, '')
      .replace(/ on[a-z]+='[^']*'/gi, '');
  }, []);

  useEffect(() => {
    const handleFileSelect = (e) => {
      if (e.target && e.target.classList.contains('diagram-upload-input')) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
          const base64 = event.target.result;
          const liveContainer = e.target.closest('.diagram-placeholder');
          const editorDiv = e.target.closest('[id^="translation-editor-"]');

          if (liveContainer && editorDiv) {
            const detachedRoot = editorDiv.cloneNode(true);
            const detachedContainer = detachedRoot.querySelector('.diagram-placeholder') || detachedRoot;
            detachedContainer.innerHTML = `<img src="${base64}" style="max-width: 100%; max-height: 400px; border-radius: 8px; display: block; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);" alt="Uploaded Diagram" />`;
            detachedContainer.style.border = 'none';
            detachedContainer.style.background = 'transparent';
            detachedContainer.style.padding = '0';

            const pageId = parseInt(editorDiv.id.replace('translation-editor-', ''), 10);

            const customEvent = new CustomEvent('diagramUploaded', {
              detail: { pageId, newHtml: detachedRoot.innerHTML }
            });
            document.dispatchEvent(customEvent);
          }
        };
        reader.readAsDataURL(file);
      }
    };

    const handleDiagramStateUpdate = (e) => {
      const { pageId, newHtml } = e.detail;
      setParsedSections(prev => prev.map(s => (s.id === pageId ? { ...s, translatedHtml: newHtml } : s)));
    };

    document.addEventListener('change', handleFileSelect);
    document.addEventListener('diagramUploaded', handleDiagramStateUpdate);

    return () => {
      document.removeEventListener('change', handleFileSelect);
      document.removeEventListener('diagramUploaded', handleDiagramStateUpdate);
    };
  }, []);

  const extractPageImageBase64 = async (pageId) => {
    if (!pdfDocRef.current) return null;
    try {
      const page = await pdfDocRef.current.getPage(pageId);
      const viewport = page.getViewport({ scale: 2.0 });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: ctx, viewport }).promise;
      return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
    } catch (e) {
      console.warn("Visual image extraction skipped for page", pageId, e);
      return null;
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setErrorMsg(null);
    setSuccessMsg(null);

    const fileName = file.name.toLowerCase();
    const isPdf = fileName.endsWith('.pdf');

    if (!isPdf) {
      setErrorMsg("Invalid format. Please upload a PDF document.");
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setErrorMsg("File size exceeds 100MB limit. Please upload a smaller document.");
      return;
    }

    if (!window.pdfjsLib) {
      setErrorMsg("PDF Engine initializing. Please try again in a few seconds.");
      return;
    }

    setFileData({ name: file.name, size: (file.size / (1024 * 1024)).toFixed(2) });
    setIsParsing(true);
    setParseProgress(0);

    try {
      const sections = [];

      if (isPdf) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        pdfDocRef.current = pdf;

        for (let i = 1; i <= pdf.numPages; i++) {
          if (i % 5 === 0) {
            setParseProgress(Math.round((i / pdf.numPages) * 100));
            await new Promise(resolve => setTimeout(resolve, 0));
          }

          const page = await pdf.getPage(i);
          const textContent = await page.getTextContent();

          const isRtlPage = sourceLangMode === 'ar_ur';
          const positionedItems = textContent.items
            .filter(item => item.str !== undefined)
            .map(item => ({
              str: item.str,
              x: item.transform ? item.transform[4] : 0,
              y: item.transform ? item.transform[5] : 0
            }));

          const lineTolerance = 3;
          positionedItems.sort((a, b) => {
            if (Math.abs(a.y - b.y) > lineTolerance) return b.y - a.y;
            return isRtlPage ? b.x - a.x : a.x - b.x;
          });

          let pageRawText = positionedItems.map(item => item.str).join(" ");

          sections.push({
            id: i,
            meta: { partName: `Page ${i}` },
            content: { rawText: pageRawText },
            translatedHtml: "",
            originalTranslatedHtml: "",
            translationStatus: 'idle',
            isPdf: true
          });
        }
      }

      setParsedSections(sections);
      if (sections.length > 0) {
        setActiveSectionId(sections[0].id);
        setSuccessMsg("Document loaded successfully! Ready for translation.");
      } else {
        setErrorMsg("No readable text found in the uploaded file.");
      }
    } catch (error) {
      console.error(error);
      setErrorMsg("Error reading the file. Please verify the document.");
    } finally {
      setIsParsing(false);
    }
  };

  const scheduleRequestSlot = () => {
    const slot = throttleRef.current.then(async () => {
      const now = Date.now();
      const wait = REQUEST_INTERVAL_MS - (now - lastRequestTimeRef.current);
      if (wait > 0) {
        await new Promise(resolve => setTimeout(resolve, wait));
      }
      lastRequestTimeRef.current = Date.now();
      try {
        localStorage.setItem(LAST_REQUEST_TIME_STORAGE_KEY, String(lastRequestTimeRef.current));
      } catch (e) { /* storage quota or unavailable - pacing still works in-memory */ }
    });
    throttleRef.current = slot.catch(() => {});
    return slot;
  };

  // ---------------------------------------------------------------------
  // callMistral: single model (ministral-14b-2512), 5-key strict round-robin.
  //
  // Rotation behavior (per spec):
  //   Key 1 works -> Key 1 hits its limit/cap -> auto-switch to Key 2 ->
  //   Key 2 hits its limit -> Key 3 -> Key 4 -> Key 5 -> back to Key 1 -> ...
  //
  // A key only rotates away on:
  //   - 429 (quota/limit reached) -> move to next key, this key stays usable
  //     again on a future full cycle (e.g. after its monthly cap resets).
  //   - 401/403 (invalid key) -> marked permanently dead for this session.
  // One full round-robin pass (every active key tried once) without success
  // throws ALL_KEYS_EXHAUSTED rather than looping forever and hammering the
  // API - the rotation pointer itself is preserved so the NEXT translation
  // call resumes the cycle from where it left off.
  // ---------------------------------------------------------------------
  const callMistral = async (content) => {
    const activeKeys = apiKeys.map(k => k.trim()).filter(Boolean);
    if (activeKeys.length === 0) {
      throw new Error("NO_API_KEY");
    }

    const acquireSlot = scheduleRequestSlot;
    const rot = rotationRef.current;

    if (rot.keyIdx >= activeKeys.length) rot.keyIdx = 0;

    let attempts = 0;
    const maxAttempts = activeKeys.length;

    while (attempts < maxAttempts) {
      while (rot.deadKeys.has(rot.keyIdx) && rot.deadKeys.size < activeKeys.length) {
        rot.keyIdx = (rot.keyIdx + 1) % activeKeys.length;
      }
      if (rot.deadKeys.size >= activeKeys.length) {
        throw new Error("ALL_KEYS_EXHAUSTED");
      }

      const activeKey = activeKeys[rot.keyIdx];

      try {
        const data = await fetchWithRetry(MISTRAL_API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${activeKey}`
          },
          body: JSON.stringify({
            model: MISTRAL_MODEL,
            messages: [{ role: 'user', content }]
          })
        }, 60000, acquireSlot);

        return data;
      } catch (error) {
        const status = error.status;
        const isQuotaError = status === 429 || /RESOURCE_EXHAUSTED|rate.?limit|quota/i.test(error.message || "");
        const isKeyError = status === 400 || status === 401 || status === 403 || /invalid_api_key|unauthorized|permission/i.test(error.message || "");

        if (isQuotaError) {
          // Rotate to next key; this key is NOT marked dead, it may recover
          // on the next full cycle (e.g. next billing period).
          rot.keyIdx = (rot.keyIdx + 1) % activeKeys.length;
          attempts += 1;
          continue;
        }

        if (isKeyError) {
          rot.deadKeys.add(rot.keyIdx);
          rot.keyIdx = (rot.keyIdx + 1) % activeKeys.length;
          attempts += 1;
          continue;
        }

        throw error;
      }
    }

    throw new Error("ALL_KEYS_EXHAUSTED");
  };

  const translatePage = async (pageText, pageId, isPdf) => {
    let imagePayload = null;
    if (isPdf) {
      const base64Image = await extractPageImageBase64(pageId);
      if (base64Image) {
        imagePayload = {
          type: 'image_url',
          image_url: { url: `data:image/jpeg;base64,${base64Image}` }
        };
      }
    }

    const langInstruction = sourceLangMode === 'ar_ur'
      ? "SOURCE LANGUAGE: Arabic / Urdu."
      : sourceLangMode === 'en'
      ? "SOURCE LANGUAGE: English."
      : "SOURCE LANGUAGE: Auto-detect (Arabic, Urdu, or English).";

    const prompt = buildTranslatePrompt(targetLang, langInstruction, pageText);

    try {
      const data = await callMistral([
        { type: 'text', text: prompt },
        ...(imagePayload ? [imagePayload] : [])
      ]);

      let rawHtml = (data.choices?.[0]?.message?.content || "").replace(/```html|```/gi, '').trim();

      if (rawHtml) {
        rawHtml = rawHtml
          .replace(/direction:\s*rtl;?/gi, 'direction: ltr;')
          .replace(/text-align:\s*right;?/gi, 'text-align: left;')
          .replace(/dir=["']?rtl["']?/gi, 'dir="ltr"');
      }

      return rawHtml;
    } catch (e) {
      console.error("Translation error:", e);
      if (e.message === 'NO_API_KEY') {
        setErrorMsg("Please enter at least one Mistral API key in Settings.");
      } else if (e.message === 'ALL_KEYS_EXHAUSTED') {
        setErrorMsg("All Mistral API keys have reached their limit or are invalid. Please add more keys in Settings, or wait for a key's limit to reset.");
      } else {
        setErrorMsg(`Translation failed: ${e.message}`);
      }
      return null;
    }
  };

  const verifyAndCorrectTranslation = async (pageText, pageId, isPdf, translatedHtmlDraft) => {
    let imagePayload = null;
    if (isPdf) {
      const base64Image = await extractPageImageBase64(pageId);
      if (base64Image) {
        imagePayload = {
          type: 'image_url',
          image_url: { url: `data:image/jpeg;base64,${base64Image}` }
        };
      }
    }

    const prompt = buildVerifyPrompt(targetLang, pageText, translatedHtmlDraft);

    try {
      const data = await callMistral([
        { type: 'text', text: prompt },
        ...(imagePayload ? [imagePayload] : [])
      ]);

      let rawText = (data.choices?.[0]?.message?.content || "").replace(/```html|```/gi, '').trim();

      if (!rawText || /^NO_CORRECTIONS_NEEDED$/i.test(rawText)) {
        return translatedHtmlDraft;
      }

      rawText = rawText
        .replace(/direction:\s*rtl;?/gi, 'direction: ltr;')
        .replace(/text-align:\s*right;?/gi, 'text-align: left;')
        .replace(/dir=["']?rtl["']?/gi, 'dir="ltr"');

      return rawText;
    } catch (e) {
      console.error("Verification error:", e);
      return translatedHtmlDraft;
    }
  };

  const startSequentialAnalysis = async () => {
    if (isTranslatingAll) return;
    setIsTranslatingAll(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const pages = [...parsedSections];
    const totalPages = pages.length;

    for (let i = 0; i < totalPages; i++) {
      const page = pages[i];

      if (page.translationStatus !== 'done') {
        setParsedSections(prev => {
          const copy = [...prev];
          copy[i].translationStatus = 'loading';
          return copy;
        });

        const htmlResult = await translatePage(page.content.rawText, page.id, page.isPdf);

        setParsedSections(prev => {
          const copy = [...prev];
          if (htmlResult) {
            copy[i].translatedHtml = htmlResult;
            if (!copy[i].originalTranslatedHtml) {
              copy[i].originalTranslatedHtml = htmlResult;
            }
            copy[i].translationStatus = 'done';
          } else {
            copy[i].translationStatus = 'error';
          }
          return copy;
        });
      }

      setProgress(Math.round(((i + 1) / totalPages) * 100));
    }

    setIsTranslatingAll(false);
    setSuccessMsg("Batch translation complete!");
  };

  const handleVerifyPage = async (pageId) => {
    const idx = parsedSections.findIndex(s => s.id === pageId);
    if (idx === -1) return;

    const page = parsedSections[idx];
    if (!page.translatedHtml) return;

    setErrorMsg(null);
    setSuccessMsg(null);

    setParsedSections(prev => {
      const copy = [...prev];
      copy[idx].translationStatus = 'verifying';
      return copy;
    });

    const finalHtml = await verifyAndCorrectTranslation(page.content.rawText, page.id, page.isPdf, page.translatedHtml);

    setParsedSections(prev => {
      const copy = [...prev];
      copy[idx].translatedHtml = finalHtml;
      if (!copy[idx].originalTranslatedHtml) {
        copy[idx].originalTranslatedHtml = finalHtml;
      }
      copy[idx].translationStatus = 'done';
      return copy;
    });
    setSuccessMsg(`Section ${pageId} verified successfully!`);
  };

  const handleResetTranslation = (pageId) => {
    setParsedSections(prev => prev.map(s => {
      if (s.id === pageId && s.originalTranslatedHtml) {
        return {
          ...s,
          translatedHtml: s.originalTranslatedHtml,
          translationStatus: 'done'
        };
      }
      return s;
    }));
    setSuccessMsg(`Section ${pageId} reset to its original translation.`);
  };

  const handleEditableContentBlur = (pageId, event) => {
    const updatedHtml = event.target.innerHTML;
    setParsedSections(prev => prev.map(s => s.id === pageId ? { ...s, translatedHtml: updatedHtml } : s));
  };

  const handleSaveSettings = () => {
    const keys = apiKeys.map(k => k.trim());
    if (!keys[0]) {
      setErrorMsg("Please enter at least the first Mistral API key.");
      return;
    }

    // Validate every non-empty key is 25-40 characters (Mistral key format).
    for (let i = 0; i < keys.length; i++) {
      if (keys[i] && (keys[i].length < MIN_MISTRAL_KEY_LENGTH || keys[i].length > MAX_MISTRAL_KEY_LENGTH)) {
        setErrorMsg(`API Key ${i + 1} looks invalid - Mistral API keys are ${MIN_MISTRAL_KEY_LENGTH}-${MAX_MISTRAL_KEY_LENGTH} characters long (got ${keys[i].length}).`);
        return;
      }
    }

    sessionStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));

    if (rememberApiKey) {
      localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
    } else {
      localStorage.removeItem(API_KEYS_STORAGE_KEY);
    }

    setApiKeys(keys);
    rotationRef.current = { keyIdx: 0, deadKeys: new Set() };
    setShowSettings(false);
    setSuccessMsg("Settings saved successfully.");
  };

  const exportHtml = async () => {
    if (isExporting) return;
    setIsExporting(true);
    setErrorMsg(null);

    const finishedTranslations = parsedSections.filter(s => s.translationStatus === 'done');
    if (finishedTranslations.length === 0) {
      setErrorMsg("No translated pages found. Please process at least one section first.");
      setIsExporting(false);
      return;
    }

    try {
      const fontFamilyCss = targetLang === 'bn'
        ? `body, .translated-section, .translated-section * { font-family: 'Kalpurush', sans-serif !important; }`
        : `body, .translated-section, .translated-section * { font-family: 'Times New Roman', serif !important; }`;
      const fontLink = targetLang === 'bn'
        ? `<link href="https://fonts.maateen.me/kalpurush/font.css" rel="stylesheet">`
        : '';

      let htmlContent = `
        <!DOCTYPE html>
        <html lang="${targetLang}">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Mirrored Translation - ${fileData?.name || 'Document'}</title>
          ${fontLink}
          <style>
            ${fontFamilyCss}
            body {
              background-color: #f8fafc;
              color: #0f172a;
              line-height: 1.8;
              padding: 2rem 1rem;
              max-width: 850px;
              margin: 0 auto;
            }
            .section-break {
              border-bottom: 2px dashed #cbd5e1;
              margin: 3rem 0;
              padding-bottom: 2rem;
              position: relative;
            }
            .section-marker {
              position: absolute;
              bottom: -10px;
              left: 50%;
              transform: translateX(-50%);
              background: #f8fafc;
              padding: 0 1rem;
              color: #94a3b8;
              font-size: 13px;
              font-weight: bold;
              font-family: sans-serif;
            }
            @media print {
              body { background: white; padding: 0; max-width: none; }
              .section-break { border: none; margin: 0; padding: 0; page-break-after: always; }
              .section-marker { display: none; }
            }
          </style>
        </head>
        <body>
      `;

      parsedSections.forEach((s) => {
        if (s.translationStatus !== 'done') return;
        htmlContent += `
          <div class="translated-section" id="section-${s.id}">
            <div dir="ltr">
              ${sanitizeHtml(s.translatedHtml)}
            </div>
          </div>
          <div class="section-break">
            <span class="section-marker">SECTION ${s.id}</span>
          </div>
        `;
      });

      htmlContent += `
        </body>
        </html>
      `;

      const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Translated_${targetLang === 'bn' ? 'Bangla' : 'English'}_${(fileData?.name || 'Document').split('.')[0]}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      setSuccessMsg("Document exported to HTML format successfully!");
    } catch (e) {
      setErrorMsg("Export failed. An error occurred while generating HTML.");
    } finally {
      setIsExporting(false);
    }
  };

  const handleCopyText = async (text, id) => {
    const tempEl = document.createElement('div');
    tempEl.innerHTML = text;
    const plainText = tempEl.innerText || tempEl.textContent || '';

    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(plainText);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = plainText;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
      setErrorMsg("Couldn't copy to clipboard. Please try selecting the text manually.");
    }
  };

  const activePage = parsedSections.find(s => s.id === activeSectionId);
  const outputFontClass = targetLang === 'bn' ? 'bangla-font' : 'english-font';
  const targetLabel = TARGET_LABELS[targetLang];

  return (
    <div className="h-screen flex flex-col bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-950">

      {/* Top Header Navigation Bar */}
      <nav className="relative z-40 bg-white border-b border-slate-200/80 px-6 py-3 flex flex-wrap items-center justify-between gap-3 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-md shadow-indigo-100">
            <BookOpen size={20} />
          </div>
          <div>
            <h1 className="text-base font-extrabold text-slate-800 tracking-tight leading-none mb-1">
              Ar/En/Ur to Bangla/English Translator
            </h1>
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-1">
              <Globe size={10} className="text-indigo-500" /> MULTILINGUAL MIRROR LAYOUT STUDIO (MINISTRAL 3 14B)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* Target Language Selector */}
          <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200">
            <button
              onClick={() => setTargetLang('bn')}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${targetLang === 'bn' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
              title="Translate to Bangla (Arabic, Urdu, or English source)"
            >
              → Bangla
            </button>
            <button
              onClick={() => setTargetLang('en')}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${targetLang === 'en' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
              title="Translate to English (Arabic or Urdu source only)"
            >
              → English
            </button>
          </div>

          {/* Source Language Selector */}
          <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200">
            <button
              onClick={() => setSourceLangMode('auto')}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${sourceLangMode === 'auto' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
              title="Auto Detect Language"
            >
              Auto
            </button>
            <button
              onClick={() => setSourceLangMode('ar_ur')}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${sourceLangMode === 'ar_ur' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
              title="Arabic & Urdu Mode (RTL)"
            >
              العربية / اردو
            </button>
            {targetLang === 'bn' && (
              <button
                onClick={() => setSourceLangMode('en')}
                className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${sourceLangMode === 'en' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
                title="English Mode (LTR)"
              >
                English
              </button>
            )}
          </div>

          {parsedSections.length > 0 && (
            <>
              <button
                onClick={startSequentialAnalysis}
                disabled={isTranslatingAll}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white rounded-xl text-xs font-bold flex items-center gap-2 disabled:opacity-50 transition-all shadow-sm shadow-indigo-200 cursor-pointer"
              >
                {isTranslatingAll ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>Analyzing ({progress}%)</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={14} />
                    <span>Translate Entire Document</span>
                  </>
                )}
              </button>
              <button
                onClick={exportHtml}
                disabled={isExporting}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-950 text-white rounded-xl text-xs font-bold flex items-center gap-2 disabled:opacity-50 transition-all shadow-sm cursor-pointer"
              >
                {isExporting ? <Loader2 size={14} className="animate-spin" /> : <FileDown size={14} />}
                <span>Export HTML</span>
              </button>
            </>
          )}

          <button
            onClick={() => setShowSettings(true)}
            className="p-2 rounded-xl text-slate-500 hover:bg-slate-100 transition-all cursor-pointer"
            title="Settings"
          >
            <Settings size={18} />
          </button>

          <button
            onClick={() => setShowHelpModal(true)}
            className="p-2 rounded-xl text-slate-500 hover:bg-slate-100 transition-all cursor-pointer"
            title="Workflow Guide"
          >
            <HelpCircle size={18} />
          </button>
        </div>
      </nav>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-100">
            <div className="flex justify-between items-center mb-4 border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
                <Settings size={18} className="text-indigo-600" /> Application Settings
              </h3>
              <button onClick={() => setShowSettings(false)} className="p-1 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer">
                <X size={18} />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-600">
              <div>
                <label className="block font-bold text-slate-700 mb-1">Mistral API Keys ({MISTRAL_MODEL})</label>
                <p className="text-[10px] text-slate-400 mb-2">
                  Key 1 is required. Keys 2-5 are optional and rotate in automatically once the earlier key hits its usage limit (e.g. its $10 cap). Rotation cycles 1 → 2 → 3 → 4 → 5 → back to 1, forever. Minimum {REQUEST_INTERVAL_MS / 1000}s between every request.
                </p>
                <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                  {apiKeys.map((k, idx) => (
                    <div key={idx}>
                      <label className="block text-[10px] font-bold text-slate-500 mb-1">
                        API Key {idx + 1} {idx === 0 ? '(Required)' : '(Optional)'}
                      </label>
                      <input
                        type="password"
                        value={k}
                        onChange={(e) => {
                          const next = [...apiKeys];
                          next[idx] = e.target.value;
                          setApiKeys(next);
                        }}
                        minLength={MIN_MISTRAL_KEY_LENGTH}
                        maxLength={MAX_MISTRAL_KEY_LENGTH}
                        placeholder={idx === 0 ? `Enter your Mistral API key (${MIN_MISTRAL_KEY_LENGTH}-${MAX_MISTRAL_KEY_LENGTH} chars)` : `Enter Mistral API key (optional, ${MIN_MISTRAL_KEY_LENGTH}-${MAX_MISTRAL_KEY_LENGTH} chars)`}
                        className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs focus:ring-2 focus:ring-indigo-200 focus:border-indigo-500 outline-none"
                      />
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-slate-400 mt-2">By default your keys are kept only for this browser tab/session and are cleared when you close it.</p>
                <label className="flex items-center gap-2 mt-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={rememberApiKey}
                    onChange={(e) => setRememberApiKey(e.target.checked)}
                    className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-200 cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-500">Remember these keys on this device (stores them in Local Storage across sessions)</span>
                </label>
              </div>

              <div>
                <label className="block font-bold text-slate-700 mb-1">Target Language</label>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs focus:ring-2 focus:ring-indigo-200 focus:border-indigo-500 outline-none"
                >
                  <option value="bn">Bangla (from Arabic, Urdu, or English)</option>
                  <option value="en">English (from Arabic or Urdu only)</option>
                </select>
              </div>

              <div>
                <label className="block font-bold text-slate-700 mb-1">Source Language Preset</label>
                <select
                  value={sourceLangMode}
                  onChange={(e) => setSourceLangMode(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs focus:ring-2 focus:ring-indigo-200 focus:border-indigo-500 outline-none"
                >
                  <option value="auto">Auto Detect Direction & Script</option>
                  <option value="ar_ur">Arabic / Urdu Mode (RTL Layout)</option>
                  {targetLang === 'bn' && <option value="en">English Mode (LTR Layout)</option>}
                </select>
              </div>
            </div>

            <div className="mt-6 pt-3 border-t border-slate-100 flex justify-end">
              <button
                onClick={handleSaveSettings}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl cursor-pointer"
              >
                Save & Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Help Workflow Modal */}
      {showHelpModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 shadow-2xl border border-slate-100">
            <div className="flex justify-between items-center mb-4 border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
                <Sparkles size={18} className="text-indigo-600" /> Multilingual Translation Guide
              </h3>
              <button onClick={() => setShowHelpModal(false)} className="p-1 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer">
                <X size={18} />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-600 leading-relaxed">
              <p>This studio translates documents from <strong>Arabic, Urdu, or English</strong> into <strong>Bangla</strong> (all three sources) or <strong>English</strong> (Arabic/Urdu sources only), while preserving the original page's visual layout - one page, one API call to Ministral 3 14B.</p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                  <span className="font-bold text-indigo-600 text-[10px] uppercase block mb-1">1. Upload</span>
                  Upload a PDF and click <strong>"Translate Entire Document"</strong>. Pages are processed one at a time.
                </div>
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                  <span className="font-bold text-emerald-600 text-[10px] uppercase block mb-1">2. Review</span>
                  Each page's 100% Mirrored Layout appears as soon as it finishes. Click directly into the text to make small manual edits any time.
                </div>
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                  <span className="font-bold text-amber-600 text-[10px] uppercase block mb-1">3. Re-translate / Verify</span>
                  Not happy with a page? Use <strong>Verify</strong> to run an audit/correction pass, or re-run translation for a fresh attempt.
                </div>
              </div>
            </div>

            <div className="mt-6 pt-3 border-t border-slate-100 flex justify-end">
              <button
                onClick={() => setShowHelpModal(false)}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-950 text-white text-xs font-bold rounded-xl cursor-pointer"
              >
                Got it, let's start!
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main App Content View */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden">

        {isParsing ? (
          <div className="flex-1 flex flex-col items-center justify-center space-y-4 max-w-4xl mx-auto w-full p-6 text-center animate-in fade-in duration-500">
            <Loader2 size={48} className="animate-spin text-indigo-600" />
            <div>
              <h3 className="text-xl font-bold text-slate-800">Reading Document...</h3>
              <p className="text-sm text-slate-500 mt-2">Extracting structure and text layer. This might take a moment for large files.</p>
            </div>
            {parseProgress > 0 && (
              <div className="w-64 mt-4 space-y-2">
                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-600 transition-all duration-300" style={{ width: `${parseProgress}%` }}></div>
                </div>
                <p className="text-xs font-bold text-slate-500">{parseProgress}% Complete</p>
              </div>
            )}
          </div>
        ) : !fileData ? (
          <div className="flex-1 max-w-4xl mx-auto w-full p-6 md:p-12 flex flex-col justify-center">
            <div className="text-center mb-8 max-w-2xl mx-auto">
              <span className="bg-indigo-50 text-indigo-700 px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest mb-4 inline-block border border-indigo-100">
                Arabic • Urdu • English → Bangla / English Mirror Studio
              </span>
              <h2 className="text-3xl md:text-5xl font-black text-slate-900 tracking-tight mb-4 leading-tight">
                Preserve Original Layouts. <br/>
                <span className="text-indigo-600">Translate to {targetLabel}.</span>
              </h2>
              <p className="text-slate-500 text-sm md:text-base leading-relaxed">
                Upload Arabic, Urdu, or English PDF books. Reconstruct visual layouts, solve page-junction text splits, and generate publication-ready {targetLabel} translations (exported as HTML), powered by Ministral 3 14B.
              </p>
            </div>

            <label className="group block w-full max-w-xl mx-auto aspect-video border-3 border-dashed border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/10 rounded-2xl bg-white transition-all cursor-pointer relative overflow-hidden shadow-xs">
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
                <div className="w-14 h-14 bg-slate-50 group-hover:bg-indigo-50 rounded-2xl flex items-center justify-center mb-4 transition-all">
                  <Upload size={24} className="text-slate-400 group-hover:text-indigo-600" />
                </div>
                <div className="space-y-2">
                  <span className="bg-slate-900 text-white px-6 py-2.5 rounded-xl font-bold text-xs block shadow-md shadow-slate-200 group-hover:bg-indigo-600 transition-colors">
                    Browse PDF File
                  </span>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Supports .pdf documents</p>
                </div>
              </div>
              <input type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>
        ) : (
          <div className="flex-1 flex flex-col md:flex-row overflow-hidden w-full">

            {/* Left Sidebar: Section / Page Selector */}
            <aside className="w-full md:w-72 bg-white border-b md:border-b-0 md:border-r border-slate-200/80 flex flex-col h-full shrink-0">

              <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                <div className="flex items-center gap-3 overflow-hidden">
                  <div className="w-8 h-8 bg-indigo-50 text-indigo-600 rounded-lg flex items-center justify-center shrink-0">
                    <FileText size={16} />
                  </div>
                  <div className="overflow-hidden">
                    <h4 className="text-xs font-bold text-slate-800 truncate" title={fileData.name}>{fileData.name}</h4>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{parsedSections.length} Sections • {fileData.size} MB</p>
                  </div>
                </div>

                <button
                  onClick={() => {
                    try { localStorage.removeItem(SESSION_STORAGE_KEY); } catch (e) { /* ignore */ }
                    window.location.reload();
                  }}
                  className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all cursor-pointer"
                  title="Close Document (clears saved session)"
                >
                  <X size={14} />
                </button>
              </div>

              <div className="px-4 py-2 bg-slate-100/50 text-slate-500 text-[10px] font-black uppercase tracking-wider border-b border-slate-100">
                Document Navigation
              </div>

              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {parsedSections.map((s) => {
                  const isActive = activeSectionId === s.id;
                  let statusBg = "bg-slate-100 text-slate-500";
                  if (isActive) statusBg = "bg-indigo-600 text-white shadow-xs";

                  return (
                    <div
                      key={s.id}
                      className={`w-full text-left px-3 py-2.5 rounded-xl flex flex-col transition-all ${isActive ? 'bg-indigo-50/80 border border-indigo-100/50 shadow-xs' : 'hover:bg-slate-50 border border-transparent'}`}
                    >
                      <div
                        onClick={() => setActiveSectionId(s.id)}
                        className="flex items-center justify-between cursor-pointer"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <span className={`w-7 h-7 flex items-center justify-center rounded-lg font-black text-xs shrink-0 ${statusBg}`}>
                            {s.id}
                          </span>
                          <div className="min-w-0">
                            <span className={`text-xs block truncate ${isActive ? 'text-indigo-950 font-bold' : 'text-slate-700 font-semibold'}`}>
                              {s.meta.partName}
                            </span>
                            <span className="text-[9px] text-slate-400 block leading-tight font-semibold">
                              {s.translationStatus === 'idle' ? 'Not Processed' : s.translationStatus === 'loading' ? 'Translating...' : s.translationStatus === 'verifying' ? 'Verifying...' : s.translationStatus === 'error' ? 'Failed' : 'Translated'}
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5">
                          {s.translationStatus === 'done' && (
                            <span className="w-2 h-2 rounded-full bg-emerald-500" title="Translated" />
                          )}
                          {(s.translationStatus === 'loading' || s.translationStatus === 'verifying') && (
                            <Loader2 size={12} className="animate-spin text-indigo-500" />
                          )}
                          {s.translationStatus === 'error' && (
                            <span className="w-2 h-2 rounded-full bg-red-500" title="Failed" />
                          )}
                          <ChevronRight size={14} className={isActive ? 'text-indigo-400' : 'text-slate-300'} />
                        </div>
                      </div>

                      {s.translationStatus !== 'idle' && (
                        <div className="mt-2.5 pl-10 pr-1 flex items-center gap-1.5">
                          {/* Verify (audit & auto-correct current translation via Ministral 3 14B, 1 API request) */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveSectionId(s.id);
                              handleVerifyPage(s.id);
                            }}
                            disabled={s.translationStatus === 'loading' || s.translationStatus === 'verifying' || !s.translatedHtml}
                            title={`Verify & auto-correct (Ministral 3 14B, 1 API request, ${REQUEST_INTERVAL_MS / 1000}s gap)`}
                            className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-indigo-600 hover:border-indigo-300 hover:bg-indigo-50 transition-all cursor-pointer disabled:opacity-50 shrink-0"
                          >
                            {s.translationStatus === 'verifying' ? (
                              <Loader2 size={12} className="animate-spin" />
                            ) : (
                              <ShieldCheck size={12} />
                            )}
                          </button>

                          {/* Reset to Original Translation - instant, no API call */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleResetTranslation(s.id);
                            }}
                            disabled={s.translationStatus === 'loading' || s.translationStatus === 'verifying' || !s.originalTranslatedHtml}
                            title="Reset to Original Translation (no API request)"
                            className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-purple-600 hover:border-purple-300 hover:bg-purple-50 transition-all cursor-pointer disabled:opacity-50 shrink-0"
                          >
                            <Sparkles size={12} />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </aside>

            {/* Main Dual Workspace Editor */}
            <div className="flex-1 flex flex-col overflow-hidden bg-slate-50">

              <div className="bg-white border-b border-slate-200/80 px-6 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-500">Workspace:</span>
                  <span className="text-xs font-extrabold text-indigo-700 bg-indigo-50 border border-indigo-100 px-2.5 py-1 rounded-md">
                    Section {activeSectionId} Workspace
                  </span>
                </div>
              </div>

              <div className="flex-1 flex flex-col overflow-hidden p-6 gap-6">

                <div className="flex-1 flex flex-col bg-white rounded-xl border border-slate-200/80 overflow-hidden shadow-xs">

                  <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                    <span className="text-xs font-extrabold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                      <Globe size={14} className="text-indigo-500" /> 100% Mirrored Target Layout ({targetLabel})
                    </span>
                    <div className="flex items-center gap-2">
                      {activePage?.translationStatus === 'done' && (
                        <>
                          <span className="text-[9px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-100 px-2 py-0.5 rounded-md flex items-center gap-1">
                            <CheckCircle2 size={10} /> Translated
                          </span>
                          <button
                            onClick={() => handleCopyText(activePage.translatedHtml, `translated-${activeSectionId}`)}
                            className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-600 transition-all cursor-pointer"
                            title="Copy Translated Text"
                          >
                            {copiedId === `translated-${activeSectionId}` ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-white">
                    {(() => {
                      if (!activePage) return null;

                      if (activePage.translationStatus === 'idle') {
                        return (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-10 h-10 bg-slate-50 text-slate-400 rounded-full flex items-center justify-center">
                              <Globe size={18} />
                            </div>
                            <div className="max-w-xs space-y-1">
                              <h4 className="text-xs font-bold text-slate-800">Not Yet Translated</h4>
                              <p className="text-[11px] text-slate-400">Click <strong>"Translate Entire Document"</strong> to generate the {targetLabel} mirrored translation for this page.</p>
                            </div>
                          </div>
                        );
                      }

                      if (activePage.translationStatus === 'loading') {
                        return (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-8 h-8 border-3 border-indigo-100 rounded-full animate-spin border-t-indigo-600"></div>
                            <div className="space-y-1">
                              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Translating to {targetLabel}...</p>
                              <p className="text-[10px] text-slate-400">Mirroring document structure, style, and terminology</p>
                            </div>
                          </div>
                        );
                      }

                      if (activePage.translationStatus === 'verifying') {
                        return (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-8 h-8 border-3 border-indigo-100 rounded-full animate-spin border-t-indigo-600"></div>
                            <div className="space-y-1">
                              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Verifying & Correcting...</p>
                              <p className="text-[10px] text-slate-400">Auditing the translation against the original source</p>
                            </div>
                          </div>
                        );
                      }

                      if (activePage.translationStatus === 'error') {
                        return (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-2">
                            <AlertCircle size={28} className="text-red-500" />
                            <div className="max-w-xs space-y-1">
                              <h4 className="text-xs font-bold text-slate-800">Translation Failed</h4>
                              <p className="text-[11px] text-slate-400">Error translating content to {targetLabel}. Please retry.</p>
                            </div>
                          </div>
                        );
                      }

                      return (
                        <div className="flex-1 flex flex-col">
                          <p className="text-[10px] text-slate-400 mb-3 border-b border-dashed border-slate-100 pb-2 flex items-center gap-1">
                            💡 <strong>Interactive Editor:</strong> Click inside to make manual edits directly - changes save automatically.
                          </p>
                          <div
                            id={`translation-editor-${activeSectionId}`}
                            contentEditable={true}
                            onBlur={(e) => handleEditableContentBlur(activeSectionId, e)}
                            className={`mirror-flow ${outputFontClass} text-lg text-slate-800 text-left outline-none ring-offset-2 focus:ring-2 focus:ring-indigo-100 rounded-xl flex-1 select-text animate-in fade-in duration-300`}
                            style={{ direction: 'ltr' }}
                            dangerouslySetInnerHTML={{ __html: sanitizeHtml(activePage.translatedHtml) }}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>

              </div>
            </div>

          </div>
        )}
      </main>

      {/* Toast Notifications */}
      {errorMsg && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-slate-900 text-white px-5 py-3.5 rounded-xl shadow-xl flex items-center gap-3 z-50 animate-in zoom-in slide-in-from-bottom-10 border border-slate-800 max-w-md w-[90%]">
          <AlertCircle className="text-red-400 shrink-0" size={18} />
          <span className="text-xs font-bold flex-1 leading-snug">{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="p-1 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white cursor-pointer">
            <X size={14} />
          </button>
        </div>
      )}

      {successMsg && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-emerald-950 text-emerald-100 px-5 py-3.5 rounded-xl shadow-xl flex items-center gap-3 z-50 animate-in zoom-in slide-in-from-bottom-10 border border-emerald-900 max-w-md w-[90%]">
          <CheckCircle2 className="text-emerald-400 shrink-0" size={18} />
          <span className="text-xs font-bold flex-1 leading-snug">{successMsg}</span>
          <button onClick={() => setSuccessMsg(null)} className="p-1 hover:bg-white/10 rounded-lg text-emerald-400 hover:text-white cursor-pointer">
            <X size={14} />
          </button>
        </div>
      )}

    </div>
  );
};

export default App;

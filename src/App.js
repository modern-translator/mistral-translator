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
  Languages,
  AlignLeft,
  AlignRight,
  Copy,
  Check,
  Eye,
  SlidersHorizontal,
  ShieldCheck,
  RotateCcw,
  Columns,
  Layers,
  ImageOff,
  ImagePlus,
  Info,
  AlertTriangle
} from 'lucide-react';

// Exponential backoff fetch implementation with timeout and abort handling.
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
        const retryAfterHeader = response.headers ? response.headers.get('Retry-After') : null;
        if (retryAfterHeader) {
          const asSeconds = Number(retryAfterHeader);
          statusError.retryAfterMs = Number.isFinite(asSeconds)
            ? asSeconds * 1000
            : Math.max(0, new Date(retryAfterHeader).getTime() - Date.now());
        }
        throw statusError;
      }
      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      lastError = error;

      if (error.status === 429 || error.status === 400 || error.status === 403) {
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

// SINGLE MODEL ENGINE - Ministral 3 14B only, called through a small CORS
// proxy (browsers cannot call api.mistral.ai directly). No model cascade:
// this is a single paid-per-token model, not a free-tier daily-cap chain.
const MISTRAL_MODEL = 'ministral-14b-2512';
const MISTRAL_API_URL = 'https://mistral-proxy-wk5t.onrender.com/api/translate';

// Exactly 5 key slots, rotated in strict round-robin order: 1 -> 2 -> 3 -> 4
// -> 5 -> back to 1 -> ... A key rotates away the moment it returns a
// quota/limit error (its cap hit) or an auth error (invalid key).
const TOTAL_API_KEY_SLOTS = 5;
// Mistral API keys are 25-40 characters long.
const MIN_MISTRAL_KEY_LENGTH = 25;
const MAX_MISTRAL_KEY_LENGTH = 40;

const MAX_RETRY_AFTER_MS = 30000;
// Minimum 2 second gap between EVERY outgoing request (first attempt or retry).
const REQUEST_INTERVAL_MS = 2100;
const LAST_REQUEST_TIME_STORAGE_KEY = 'translator_last_request_time_v1';

// PHASE 1: lightweight, non-destructive validation of a translation result.
const validateTranslationResult = (html, sourceText) => {
  if (!html || !html.trim()) {
    return { valid: false, reason: 'Empty translation result.' };
  }
  const stripped = html.replace(/<[^>]*>/g, '').trim();
  const isBlankPageMessage = /No Text Found/i.test(stripped);
  if (!isBlankPageMessage) {
    if (stripped.length < 3) {
      return { valid: false, reason: 'Translation result has no readable content.' };
    }
    const sourceLen = (sourceText || '').trim().length;
    if (sourceLen > 200 && stripped.length < 15) {
      return { valid: false, reason: 'Translation result is suspiciously short relative to the source page.' };
    }
  }

  const blockIdMatches = [...html.matchAll(/data-block-id="([^"]*)"/g)].map(m => m[1]);
  if (blockIdMatches.length > 0) {
    const seen = new Set();
    for (const id of blockIdMatches) {
      if (!id) {
        return { valid: false, reason: 'Missing block ID found in structured output.' };
      }
      if (seen.has(id)) {
        return { valid: false, reason: `Duplicate block ID "${id}" found in structured output.` };
      }
      seen.add(id);
    }
  }

  return { valid: true, reason: null };
};

// PHASE 4: lightweight, display-only diagnostics derived from existing Phase 1/2/3 data.
// This NEVER mutates translation content and NEVER blocks anything - purely informational.
const computeDiagnostics = (page) => {
  if (!page) return null;
  const structure = page.structure || null;
  const sourceBlockCount = structure && Array.isArray(structure.blocks) ? structure.blocks.length : 0;
  const html = page.translatedHtml || '';
  const translatedBlockMatches = [...html.matchAll(/data-block-id="([^"]*)"/g)];
  const translatedBlockCount = translatedBlockMatches.length;
  const visualAssetCount = structure && Array.isArray(structure.visualAssets) ? structure.visualAssets.length : 0;
  const placeholderCount = (html.match(/diagram-placeholder/g) || []).length;
  const footnoteWarnings = structure && structure.footnotes && Array.isArray(structure.footnotes.warnings)
    ? structure.footnotes.warnings
    : [];

  const warnings = [];

  if (page.translationStatus === 'done') {
    const validation = validateTranslationResult(html, page.content?.rawText);
    if (!validation.valid) {
      warnings.push({ type: 'short_or_empty', message: validation.reason });
    }
  }

  if (sourceBlockCount > 0 && translatedBlockCount > 0 && translatedBlockCount !== sourceBlockCount) {
    warnings.push({
      type: 'block_mismatch',
      message: `Source detected ${sourceBlockCount} block(s) but translated output tagged ${translatedBlockCount}.`
    });
  }

  footnoteWarnings.forEach(fw => {
    if (fw.type === 'duplicate_marker') {
      warnings.push({ type: 'footnote_duplicate', message: `Duplicate footnote marker "${fw.marker}" detected (x${fw.count}).` });
    } else if (fw.type === 'missing_marker') {
      warnings.push({ type: 'footnote_missing', message: 'A footnote block has no detectable marker.' });
    }
  });

  if (visualAssetCount > placeholderCount && page.translationStatus === 'done') {
    warnings.push({
      type: 'visual_uncertain',
      message: `Detected ${visualAssetCount} visual element(s) on the source page but only ${placeholderCount} placeholder(s) in the translation.`
    });
  }

  // Heuristic-only, non-blocking check for likely un-translated leftover script
  // (outside any explicitly marked dir="rtl" embedded span, which is allowed).
  if (page.translationStatus === 'done' && html) {
    const withoutRtlSpans = html.replace(/<[^>]*dir=["']rtl["'][^>]*>[\s\S]*?<\/[a-zA-Z0-9]+>/g, '');
    const hasArabicScript = /[\u0600-\u06FF]/.test(withoutRtlSpans.replace(/<[^>]*>/g, ''));
    if (hasArabicScript) {
      warnings.push({ type: 'possible_leftover_script', message: 'Possible untranslated Arabic/Urdu script detected outside a marked embedded span.' });
    }
  }

  return {
    sourceBlockCount,
    translatedBlockCount,
    visualAssetCount,
    placeholderCount,
    verificationStatus: page.translationStatus === 'done' ? (page.originalTranslatedHtml && page.originalTranslatedHtml !== page.translatedHtml ? 'edited-since-verify' : 'not-verified-yet') : page.translationStatus,
    warnings
  };
};

// ==================================================================
// PHASE 2: PDF EXTRACTION & STRUCTURE METADATA (unchanged from baseline)
// ==================================================================
const groupItemsIntoLines = (positionedItems, lineTolerance = 3) => {
  if (!positionedItems || positionedItems.length === 0) return [];

  const sorted = [...positionedItems].sort((a, b) => {
    if (Math.abs(a.y - b.y) > lineTolerance) return b.y - a.y;
    return a.x - b.x;
  });

  const rawLines = [];
  let current = null;
  for (const item of sorted) {
    if (!current || Math.abs(item.y - current.y) > lineTolerance) {
      current = { y: item.y, items: [item] };
      rawLines.push(current);
    } else {
      current.items.push(item);
      current.y = (current.y * (current.items.length - 1) + item.y) / current.items.length;
    }
  }

  return rawLines.map((line, idx) => {
    const orderedItems = [...line.items].sort((a, b) => a.x - b.x);
    const xs = orderedItems.map(i => i.x);
    const rights = orderedItems.map(i => i.x + (i.width || 0));
    const minX = Math.min(...xs);
    const maxX = Math.max(...rights);
    const maxHeight = Math.max(...orderedItems.map(i => i.height || 0), 0);
    const avgFontSize = orderedItems.reduce((sum, i) => sum + (i.fontSize || 0), 0) / (orderedItems.length || 1);
    return {
      lineIndex: idx,
      y: line.y,
      x: minX,
      width: Math.max(0, maxX - minX),
      height: maxHeight,
      fontSize: avgFontSize || null,
      fontName: orderedItems[0] ? orderedItems[0].fontName : null,
      text: orderedItems.map(i => i.str).join(' ').replace(/\s+/g, ' ').trim(),
      items: orderedItems
    };
  }).filter(l => l.text.length > 0);
};

const detectColumns = (lines, pageWidth) => {
  const fallback = { columnCount: 1, columns: [{ xStart: 0, xEnd: pageWidth || 0 }], confidence: 'default' };
  if (!lines || lines.length === 0 || !pageWidth) return fallback;

  const narrowLines = lines.filter(l => l.width > 0 && l.width < pageWidth * 0.55);
  if (narrowLines.length < 8) return fallback;

  const binSize = pageWidth / 20;
  const bins = {};
  narrowLines.forEach(l => {
    const bin = Math.round(l.x / binSize);
    bins[bin] = (bins[bin] || 0) + 1;
  });

  const sortedBins = Object.keys(bins).map(Number).sort((a, b) => a - b);
  const clusters = [];
  let currentCluster = [sortedBins[0]];
  for (let i = 1; i < sortedBins.length; i++) {
    if (sortedBins[i] - currentCluster[currentCluster.length - 1] <= 2) {
      currentCluster.push(sortedBins[i]);
    } else {
      clusters.push(currentCluster);
      currentCluster = [sortedBins[i]];
    }
  }
  clusters.push(currentCluster);

  const clusterInfo = clusters
    .map(c => ({
      xStart: Math.min(...c) * binSize,
      count: c.reduce((sum, b) => sum + bins[b], 0)
    }))
    .filter(c => c.count >= 3)
    .sort((a, b) => a.xStart - b.xStart);

  if (clusterInfo.length < 2 || clusterInfo.length > 4) return fallback;

  for (let i = 1; i < clusterInfo.length; i++) {
    if (clusterInfo[i].xStart - clusterInfo[i - 1].xStart < pageWidth * 0.15) {
      return fallback;
    }
  }

  const columns = clusterInfo.map((c, i) => ({
    xStart: c.xStart,
    xEnd: i < clusterInfo.length - 1 ? clusterInfo[i + 1].xStart : pageWidth
  }));

  return { columnCount: columns.length, columns, confidence: 'detected' };
};

const buildColumnReadingOrder = (lines, columnInfo, isRtlPage) => {
  if (!columnInfo || columnInfo.columnCount <= 1) {
    return [...lines].sort((a, b) => b.y - a.y);
  }
  const columns = isRtlPage ? [...columnInfo.columns].reverse() : [...columnInfo.columns];
  const assigned = columns.map(() => []);
  const unassigned = [];

  lines.forEach(line => {
    const center = line.x + line.width / 2;
    const colIdx = columns.findIndex(c => center >= c.xStart && center < c.xEnd);
    if (colIdx === -1) {
      unassigned.push(line);
    } else {
      assigned[colIdx].push(line);
    }
  });

  assigned.forEach(col => col.sort((a, b) => b.y - a.y));
  unassigned.sort((a, b) => b.y - a.y);

  const ordered = [];
  ordered.push(...unassigned.filter(l => l.width >= (columnInfo.columns[0]?.xEnd || 0) * 0.85));
  const remainderUnassigned = unassigned.filter(l => !ordered.includes(l));
  return [...ordered, ...assigned.flat(), ...remainderUnassigned];
};

const classifyBlocks = (lines, pageId, pageHeight, avgFontSize) => {
  const blocks = [];
  let blockCounter = 0;
  const makeId = () => `page-${pageId}-block-${String(++blockCounter).padStart(2, '0')}`;

  lines.forEach((line, idx) => {
    const text = line.text;
    if (!text) return;

    const fontSize = line.fontSize || null;
    const relY = pageHeight ? (pageHeight - line.y) / pageHeight : null;
    const isNearTop = relY !== null && relY < 0.07;
    const isNearBottom = relY !== null && relY > 0.88;
    const isLarger = fontSize && avgFontSize && fontSize > avgFontSize * 1.15;
    const isSmaller = fontSize && avgFontSize && fontSize < avgFontSize * 0.82;

    const looksLikePageNumber = /^[\s\d٠-٩۰-۹]{1,6}$/.test(text) && text.replace(/\s/g, '').length <= 4;
    const looksLikeFootnoteMarker = /^[\d١-٩۱-۹]{1,3}[.\)]\s/.test(text) || /^[*†‡]\s/.test(text);
    const looksLikeListItem = /^[\-•●○◦‣]\s/.test(text) || /^\(?[\dأ-يa-zA-Z]{1,3}[.\)]\s/.test(text);
    const looksLikeQuote = /^[«"“]/.test(text) || /[»"”]$/.test(text);
    const looksLikeReligiousQuote = looksLikeQuote && /[﴾﴿]/.test(text);
    const looksLikeCitation = /\([^()]{2,40},\s?\d{4}\)/.test(text) || /\[\d{1,3}\]\s*$/.test(text);

    let type = 'paragraph';
    if (looksLikePageNumber && (isNearTop || isNearBottom)) {
      type = 'page_number';
    } else if (isNearTop && idx === 0) {
      type = 'header';
    } else if (isNearBottom && idx === lines.length - 1) {
      type = 'footer';
    } else if (isSmaller && isNearBottom) {
      type = 'footnote';
    } else if (looksLikeReligiousQuote) {
      type = 'religious_quote';
    } else if (looksLikeQuote) {
      type = 'quote';
    } else if (looksLikeCitation) {
      type = 'citation';
    } else if (isLarger) {
      type = 'heading';
    } else if (looksLikeListItem) {
      type = 'list_item';
    }

    blocks.push({
      id: makeId(),
      type,
      text,
      x: line.x,
      y: line.y,
      width: line.width,
      height: line.height,
      fontSize,
      hasFootnoteMarker: looksLikeFootnoteMarker
    });
  });

  return blocks;
};

const detectTableCandidates = (lines, pageWidth) => {
  if (!lines || lines.length < 3 || !pageWidth) return [];
  const candidates = [];
  let runStart = null;

  const isRowLike = (line) => line.items.length >= 3 && line.width < pageWidth * 0.9;

  for (let i = 0; i < lines.length; i++) {
    if (isRowLike(lines[i])) {
      if (runStart === null) runStart = i;
    } else if (runStart !== null) {
      if (i - runStart >= 3) candidates.push(lines.slice(runStart, i));
      runStart = null;
    }
  }
  if (runStart !== null && lines.length - runStart >= 3) {
    candidates.push(lines.slice(runStart, lines.length));
  }

  return candidates.map((rows, tIdx) => {
    const colCount = Math.min(...rows.map(r => r.items.length));
    return {
      tableIndex: tIdx,
      rowCount: rows.length,
      columnCount: colCount,
      confidence: rows.length >= 4 && colCount >= 3 ? 'medium' : 'low',
      cells: rows.map(r => r.items.slice(0, colCount).map(i => i.str))
    };
  }).filter(t => t.confidence !== 'low' || t.rowCount >= 3);
};

const buildFootnoteMetadata = (blocks) => {
  const footnoteBlocks = blocks.filter(b => b.type === 'footnote');
  const links = [];
  const markerCounts = {};

  footnoteBlocks.forEach(fb => {
    const m = fb.text.match(/^([\d١-٩۱-۹]{1,3}|[*†‡])[.\)]?\s/);
    const marker = m ? m[1] : null;
    if (marker) {
      markerCounts[marker] = (markerCounts[marker] || 0) + 1;
      links.push({ footnoteBlockId: fb.id, marker });
    } else {
      links.push({ footnoteBlockId: fb.id, marker: null });
    }
  });

  const warnings = [];
  Object.entries(markerCounts).forEach(([marker, count]) => {
    if (count > 1) {
      warnings.push({ type: 'duplicate_marker', marker, count });
    }
  });
  links.filter(l => !l.marker).forEach(l => {
    warnings.push({ type: 'missing_marker', footnoteBlockId: l.footnoteBlockId });
  });

  return { links, warnings };
};

// v6 FIX 8: rough estimate of body-paragraph count from vertical line spacing,
// so the main translation prompt can be given a concrete checkable number
// ("output approximately N paragraphs") instead of only a vague "match the
// source" instruction. This is a best-effort estimate from spacing, not a
// guaranteed-exact count - consecutive 'paragraph'-type lines with a
// larger-than-typical vertical gap between them are treated as a paragraph
// break.
const estimateBodyParagraphCount = (structure) => {
  const paraBlocks = (structure?.blocks || [])
    .filter(b => b.type === 'paragraph' && typeof b.y === 'number')
    .sort((a, b) => b.y - a.y); // PDF y increases upward - descending y = reading top to bottom

  if (paraBlocks.length === 0) return null;

  let count = 1;
  for (let i = 1; i < paraBlocks.length; i++) {
    const gap = paraBlocks[i - 1].y - paraBlocks[i].y;
    const typicalLineHeight = paraBlocks[i].height || paraBlocks[i].fontSize || 14;
    if (gap > typicalLineHeight * 1.6) count += 1;
  }
  return count;
};

// =============================================================================
// v7 TWO-PHASE PIPELINE - ASSEMBLY ENGINE (pure code, zero AI judgment)
//
// Phase A ("Extraction") decides EVERYTHING about a block's structure/appearance
// (type, alignment, size tier, color, where emphasis/footnote-refs sit within
// the text) and returns it as JSON, in the ORIGINAL language.
// Phase B ("Translation") receives ONLY {id, text} pairs for translatable
// blocks and returns ONLY {id, translatedText} - it never sees or decides
// alignment/size/color/type, and it must preserve two literal token types
// verbatim (untranslated, unmoved) rather than deciding formatting itself:
//   - a footnote-reference token marking exactly where an inline superscript
//     marker belongs within body text
//   - a highlight-span token marking exactly which words are emphasized
// This function (assemblePage) then combines Phase A's locked structure with
// Phase B's translated words - using the SAME fixed style rules every time,
// in a SAME fixed element order every time - to produce the final HTML. No
// call, AI or otherwise, ever decides formatting at assembly time.
// =============================================================================

const FN_TOKEN_RE = /⟦FN:([^⟧]{1,6})⟧/g;
const HL_TOKEN_RE = /⟦HL⟧([\s\S]*?)⟦\/HL⟧/g;

// Converts Phase A's inline structural tokens (already present, verbatim, in
// either the original text or Phase B's translated text - both are handled
// identically here) into real HTML. This is the ONLY place these tokens ever
// become actual <sup>/<span> markup - neither Phase A nor Phase B outputs
// real HTML tags inside block text, only these plain-text placeholder tokens,
// which keeps Phase B's literal find/replace job unambiguous.
const applyInlineTokens = (text) => {
  if (!text) return text;
  return text
    .replace(HL_TOKEN_RE, '<span style="color: #BE123C; font-weight: bold;">$1</span>')
    .replace(FN_TOKEN_RE, '<sup>$1</sup>');
};

const SIZE_TIER_PX = {
  large_heading: '27px',
  medium_heading: '21px',
  subheading: '18px',
  body: '17px',
  small: '14px'
};

const escapeAttr = (s) => String(s || '').replace(/"/g, '&quot;');

// Renders exactly ONE block's HTML from Phase A's locked data + its final
// text (Phase B's translation for translatable blocks, or Phase A's own
// original-language text unchanged for preserve=true blocks like URLs/icons/
// preserved book titles). Every style decision here comes from `block` -
// never re-judged, never guessed, always the same mapping.
const renderBlockHtml = (block, finalText, targetLang) => {
  const align = block.align || 'left';
  const sizePx = SIZE_TIER_PX[block.sizeTier] || SIZE_TIER_PX.body;
  const color = block.color || null;
  const idAttr = block.id ? ` data-block-id="${escapeAttr(block.id)}"` : '';
  const content = applyInlineTokens(finalText || '');

  if (block.type === 'image') {
    return `<div class="diagram-placeholder" style="border: 2px dashed #CBD5E1; padding: 20px; text-align: center; border-radius: 8px; margin: 16px 0; cursor: pointer;"${idAttr}><p style="font-size: 14px; color: #64748B; margin-bottom: 8px;">📤 Image Detected. Click here to upload replacement.</p></div>`;
  }

  if (block.type === 'heading') {
    return `<p style="text-align: ${align}; color: ${color || '#4338CA'}; font-size: ${sizePx}; font-weight: bold; margin-bottom: 20px;"${idAttr}>${content}</p>`;
  }
  if (block.type === 'subheading') {
    return `<p style="text-align: ${align}; color: ${color || '#0F172A'}; font-size: ${sizePx}; font-weight: bold; margin-bottom: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;"${idAttr}>${content}</p>`;
  }
  if (block.type === 'url') {
    return `<p style="text-align: ${align}; color: ${color || '#334155'}; font-size: ${sizePx}; line-height: 1.8; margin-bottom: 16px;"${idAttr}><a href="${escapeAttr(finalText)}" style="color: #2563EB; text-decoration: underline;">${escapeAttr(finalText)}</a></p>`;
  }
  if (block.type === 'icon') {
    return `<p style="text-align: ${align}; font-size: ${sizePx}; margin-bottom: 8px;"${idAttr}>${escapeAttr(finalText)}</p>`;
  }
  // paragraph, quote, citation, list_item all share the body template
  return `<p style="text-align: ${align}; color: ${color || '#334155'}; font-size: ${sizePx}; line-height: 1.8; margin-bottom: 16px;"${idAttr}>${content}</p>`;
};

// Assembles the full page from Phase A's locked structure + Phase B's
// translation map ({blockId: translatedText}). Element ORDER here is fixed
// by this function, never by an AI's placement judgment - header first, body
// blocks in their given order, all footnotes collected into ONE block,
// footer, then page number, always in that sequence. This is what makes
// "floating page number" / "misplaced footnotes" structurally impossible
// rather than merely instructed against.
const assemblePage = (extraction, translationMap, targetLang) => {
  if (!extraction) return '';

  const textFor = (block) => {
    if (block.preserve) return block.text || '';
    return translationMap[block.id] ?? block.text ?? '';
  };

  const headerHtml = (extraction.header?.lines || [])
    .map(line => `<p style="text-align: ${line.align || 'left'}; color: #94A3B8; font-size: ${SIZE_TIER_PX[line.sizeTier] || SIZE_TIER_PX.small}; margin-bottom: 4px;">${applyInlineTokens(translationMap[`header:${line.id}`] ?? line.text ?? '')}</p>`)
    .join('');

  const bodyBlocks = (extraction.blocks || []).filter(b => b.type !== 'footnote');
  const bodyHtml = bodyBlocks.map(b => renderBlockHtml(b, textFor(b), targetLang)).join('');

  const footnoteBlocks = (extraction.blocks || []).filter(b => b.type === 'footnote');
  const footnotesHtml = footnoteBlocks.length > 0
    ? `<div class="footnotes" style="margin-top: 24px; border-top: 1px solid #E2E8F0; padding-top: 12px;">${footnoteBlocks.map(b => `<p style="text-align: left; color: #64748B; font-size: 14px;"><sup>${escapeAttr(b.marker || '')}</sup> ${applyInlineTokens(textFor(b))}</p>`).join('')}</div>`
    : '';

  const footerHtml = (extraction.footer?.lines || [])
    .map(line => `<p style="text-align: ${line.align || 'center'}; color: #94A3B8; font-size: ${SIZE_TIER_PX[line.sizeTier] || SIZE_TIER_PX.small}; margin-top: 4px;">${applyInlineTokens(translationMap[`footer:${line.id}`] ?? line.text ?? '')}</p>`)
    .join('');

  const pageNumberHtml = extraction.pageNumber?.text
    ? `<p style="text-align: ${extraction.pageNumber.align || 'center'}; color: #94A3B8; font-size: 13px; margin-top: 8px;">${escapeAttr(extraction.pageNumber.text)}</p>`
    : '';

  return headerHtml + bodyHtml + footnotesHtml + footerHtml + pageNumberHtml;
};

// Robust JSON extraction from a model response - strips markdown fences, and
// if the model wrapped valid JSON in any commentary, extracts the outermost
// {...} object instead of failing outright.
const parseJsonFromModelResponse = (rawText) => {
  if (!rawText) return null;
  let cleaned = rawText.replace(/```json|```/gi, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      try {
        return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1));
      } catch (e2) {
        return null;
      }
    }
    return null;
  }
};

const detectLikelyContinuation = (blocks) => {
  const contentBlocks = blocks.filter(b => !['header', 'footer', 'page_number'].includes(b.type));
  if (contentBlocks.length === 0) return { endsWithContinuation: false, lastBlockId: null };
  const last = contentBlocks[contentBlocks.length - 1];
  const trimmed = (last.text || '').trim();
  const endsWithTerminalPunctuation = /[.!?؟۔।:؛]\s*$/.test(trimmed);
  return {
    endsWithContinuation: trimmed.length > 0 && !endsWithTerminalPunctuation,
    lastBlockId: last.id
  };
};

const extractVisualAssetMetadata = async (page, pageId) => {
  try {
    if (!page.getOperatorList || !window.pdfjsLib || !window.pdfjsLib.OPS) return [];
    const opList = await page.getOperatorList();
    const OPS = window.pdfjsLib.OPS;
    const imageOps = new Set([OPS.paintImageXObject, OPS.paintJpegXObject, OPS.paintImageMaskXObject].filter(Boolean));
    const assets = [];
    let counter = 0;
    for (let i = 0; i < opList.fnArray.length; i++) {
      if (imageOps.has(opList.fnArray[i])) {
        counter += 1;
        assets.push({
          assetId: `page-${pageId}-visual-${counter}`,
          type: 'image',
          pageId,
          approxX: null,
          approxY: null,
          width: null,
          height: null,
          description: null
        });
      }
    }
    return assets;
  } catch (e) {
    console.warn('PHASE 2: visual asset metadata scan skipped for page', pageId, e);
    return [];
  }
};

const computeRepeatedHeaderFooterMetadata = (allPageStructures) => {
  const normalize = (t) => (t || '').trim().toLowerCase().replace(/\s+/g, ' ').replace(/[\d٠-٩۰-۹]+/g, '#');
  const counts = {};

  allPageStructures.forEach(ps => {
    (ps.blocks || []).forEach(b => {
      if (b.type === 'header' || b.type === 'footer') {
        const key = normalize(b.text);
        if (key.length < 3) return;
        counts[key] = (counts[key] || 0) + 1;
      }
    });
  });

  return allPageStructures.map(ps => {
    const repeated = (ps.blocks || [])
      .filter(b => b.type === 'header' || b.type === 'footer')
      .map(b => {
        const key = normalize(b.text);
        return { blockId: b.id, occurrences: counts[key] || 1 };
      })
      .filter(r => r.occurrences > 1);
    return { ...ps, repeatedHeaderFooter: repeated };
  });
};

// ==================================================================
// PHASE 3: TRANSLATION PROMPT & CONSISTENCY HELPERS
// ==================================================================

// PHASE 3: normalize a term for glossary keying (case/space-insensitive match)
const normalizeGlossaryKey = (s) => (s || '').trim().toLowerCase().replace(/\s+/g, ' ');

// PHASE 3: parses an optional trailing "GLOSSARY_JSON:[...]" line that the
// model is asked to append after the HTML. Never throws - any parse failure
// is treated as "no suggestions this time" so translation output is never
// blocked or corrupted by a glossary-formatting slip.
const parseGlossarySuggestionsFromResponse = (rawText) => {
  if (!rawText) return { cleanedHtml: rawText, suggestions: [] };
  const marker = /GLOSSARY_JSON\s*:\s*(\[[\s\S]*?\])\s*$/i;
  const match = rawText.match(marker);
  if (!match) return { cleanedHtml: rawText, suggestions: [] };

  const cleanedHtml = rawText.slice(0, match.index).trim();
  let suggestions = [];
  try {
    const parsed = JSON.parse(match[1]);
    if (Array.isArray(parsed)) {
      suggestions = parsed
        .filter(e => e && typeof e.source === 'string' && typeof e.translated === 'string' && e.source.trim() && e.translated.trim())
        .slice(0, 8)
        .map(e => ({ source: e.source.trim(), translated: e.translated.trim() }));
    }
  } catch (e) {
    console.warn('PHASE 3: glossary suggestion block could not be parsed, skipping.', e);
  }
  return { cleanedHtml, suggestions };
};

// PHASE 3: merges newly-suggested glossary entries into the running
// document-level glossary. Existing entries (especially anything a future
// UI might mark userConfirmed) are never silently overwritten in meaning -
// only occurrence counts increase and the first-seen Bangla rendering is
// kept, which is what gives later pages a stable, consistent term to reuse.
const mergeGlossaryEntries = (currentList, suggestions) => {
  if (!suggestions || suggestions.length === 0) return currentList;
  const next = [...currentList];
  suggestions.forEach(({ source, translated }) => {
    const key = normalizeGlossaryKey(source);
    if (!key) return;
    const existingIdx = next.findIndex(e => normalizeGlossaryKey(e.source) === key);
    if (existingIdx === -1) {
      next.push({ source, translated, occurrences: 1, userConfirmed: false });
    } else {
      next[existingIdx] = {
        ...next[existingIdx],
        occurrences: (next[existingIdx].occurrences || 1) + 1
        // The translated rendering is intentionally left as-is once first recorded,
        // unless a future UI marks it userConfirmed with an explicit edit.
      };
    }
  });
  return next;
};

// PHASE 3: small, targeted glossary excerpt for the prompt - top entries by
// occurrence, capped so token usage stays bounded regardless of document size.
const buildGlossaryPromptExcerpt = (glossaryList, limit = 25) => {
  if (!glossaryList || glossaryList.length === 0) return null;
  const top = [...glossaryList]
    .sort((a, b) => (b.occurrences || 0) - (a.occurrences || 0))
    .slice(0, limit);
  if (top.length === 0) return null;
  return top.map(e => `- "${e.source}" => "${e.translated}"`).join('\n');
};

// PHASE 3: block-to-translation correspondence prompt section. Only used
// when Phase 2 produced structured blocks for this page; falls back to
// nothing (whole-page prompt behavior) otherwise.
const buildBlockMapPromptSection = (structure) => {
  if (!structure || !Array.isArray(structure.blocks) || structure.blocks.length === 0) return null;
  const rows = structure.blocks.map(b => {
    const snippet = (b.text || '').slice(0, 90).replace(/\s+/g, ' ');
    return `- id="${b.id}" type="${b.type}": "${snippet}${b.text.length > 90 ? '…' : ''}"`;
  }).join('\n');
  return rows;
};

// PHASE 3: small, targeted document-level context (neighboring-page
// continuation + a pseudo section heading). Deliberately excludes full page
// text from unrelated parts of the document.
const buildContextPromptSection = (structure, prevPageStructure) => {
  const lines = [];
  const currentHeadingBlock = structure?.blocks?.find(b => b.type === 'heading');
  if (currentHeadingBlock) {
    lines.push(`This page's likely section/heading: "${currentHeadingBlock.text.slice(0, 100)}"`);
  }
  if (prevPageStructure?.continuation?.endsWithContinuation) {
    lines.push('The previous page ended mid-sentence (no terminal punctuation). If this page\'s first paragraph is a continuation of that sentence, translate it as the continuation it is - do not insert an artificial new-sentence break, and do not fabricate or repeat the missing words from the previous page.');
  }
  const prevHeadingBlock = prevPageStructure?.blocks?.find(b => b.type === 'heading');
  if (prevHeadingBlock && !currentHeadingBlock) {
    lines.push(`The previous page's section/heading was: "${prevHeadingBlock.text.slice(0, 100)}" - this page may still be under that same section.`);
  }
  if (lines.length === 0) return null;
  return lines.join('\n');
};

// PHASE 3: repeated header/footer consistency hint - looks up any existing
// glossary translation for text Phase 2 flagged as repeating elsewhere in
// the document, so the same running header/footer isn't re-worded every time.
const buildRepeatedMetadataPromptSection = (structure, glossaryList) => {
  if (!structure || !Array.isArray(structure.repeatedHeaderFooter) || structure.repeatedHeaderFooter.length === 0) return null;
  if (!glossaryList || glossaryList.length === 0) return null;
  const hints = [];
  structure.repeatedHeaderFooter.forEach(r => {
    const block = structure.blocks.find(b => b.id === r.blockId);
    if (!block) return;
    const key = normalizeGlossaryKey(block.text);
    const match = glossaryList.find(e => normalizeGlossaryKey(e.source) === key);
    if (match) {
      hints.push(`- "${block.text.slice(0, 90)}" => use exactly: "${match.translated}"`);
    }
  });
  if (hints.length === 0) return null;
  return hints.join('\n');
};

// Display name for each supported target language.
const TARGET_LABELS = { bn: 'Bangla', en: 'English' };

// PHASE 3: static numeral-handling instructions, made deliberate rather than
// a blanket "convert every number" rule. Parameterized by target language:
// Bangla has its own numeral glyphs to convert to; English uses standard
// Western numerals, so the guidance differs slightly (see the two branches).
const buildNumeralHandlingInstructions = (targetCode) => {
  if (targetCode === 'en') {
    return `
      --- NUMERAL HANDLING (BE DELIBERATE, NOT BLANKET) ---
      English uses standard Western numerals (0-9) for all numbers, so no separate numeral-script conversion is needed the way it would be for a language with its own numeral glyphs. Still distinguish two categories of numbers before rendering them, since the source may use Arabic-Indic digits (٠١٢٣...) or Eastern numeral forms that must be normalized to standard Western digits either way:
      1. NARRATIVE NUMBERS - numbers that are part of the flowing sentence content (counts, ages, general quantities, years mentioned in prose, etc.). Render these as standard Western numerals (0-9), spelled out or as digits per normal English prose convention for the context.
      2. REFERENCE NUMBERS - footnote markers, citation numbers, page numbers/references, volume/issue numbers, ISBNs, verse/hadith numbering used as an identifier, and dates used as a formal reference (e.g. a publication date in a citation). These function as identifiers/lookup keys, not narrative content - render them as standard Western digits but NEVER change, renumber, or reformat the actual value, so they still match their referent (e.g. the footnote list, the cited volume) exactly.
      When Phase 2 structural metadata marks a block as a footnote or citation (see SOURCE BLOCK MAP below, when provided), treat its leading marker number as a REFERENCE NUMBER. When in doubt for an ambiguous number, prefer preserving it over converting it, since an incorrectly converted reference number breaks the reader's ability to find what it points to.
`;
  }
  return `
      --- NUMERAL HANDLING (BE DELIBERATE, NOT BLANKET) ---
      Distinguish two categories of numbers before deciding whether to localize them to Bangla numerals:
      1. NARRATIVE NUMBERS - numbers that are part of the flowing sentence content (counts, ages, general quantities, years mentioned in prose, etc.). Convert these to Bangla numerals as before.
      2. REFERENCE NUMBERS - footnote markers, citation numbers, page numbers/references, volume/issue numbers, ISBNs, verse/hadith numbering used as an identifier, and dates used as a formal reference (e.g. a publication date in a citation). These function as identifiers/lookup keys, not narrative content - preserve their original digit form so they still match their referent (e.g. the footnote list, the cited volume) instead of converting them.
      When Phase 2 structural metadata marks a block as a footnote or citation (see SOURCE BLOCK MAP below, when provided), treat its leading marker number as a REFERENCE NUMBER. When in doubt for an ambiguous number, prefer preserving it over converting it, since an incorrectly converted reference number breaks the reader's ability to find what it points to.
`;
};

// PHASE 3: mixed RTL/LTR handling instructions - replaces the previous blind
// "force everything to LTR" post-processing with explicit, scoped guidance.
// Parameterized by target language name for the prose wording only; the
// actual rule (page stays LTR, narrow embedded-quotation exception) is the
// same regardless of target.
const buildMixedDirectionInstructions = (targetCode) => {
  const targetName = TARGET_LABELS[targetCode];
  return `
      --- MIXED DIRECTION TEXT (LOCAL RTL SPANS) ---
      The overall page output stays LTR (${targetName} reads left-to-right) - keep outer <p>/<div> containers and their direction as specified below. However, if (per the Quranic-verse/Arabic-term handling elsewhere in this prompt) a short embedded original-script Arabic/Urdu excerpt is legitimately preserved alongside its ${targetName} rendering, wrap ONLY that embedded excerpt in its own inline element styled with dir="rtl" and style="direction: rtl; unicode-bidi: embed;" so it displays correctly in its own script, while the surrounding ${targetName} sentence and the page as a whole remain LTR. Never set rtl direction on an outer paragraph, heading, or page-level container - only on the specific embedded original-script span, and never as a substitute for translating the surrounding sentence into ${targetName}.
`;
};

// PHASE 4: block-type -> accent color mapping, used only for the optional
// diagnostics legend and for the best-effort visual accent pass below. This
// does not alter translation content; it is a read-only presentation aid.
const BLOCK_TYPE_ACCENTS = {
  heading: { label: 'Heading', color: '#4338CA', bg: '#EEF2FF' },
  header: { label: 'Header', color: '#64748B', bg: '#F1F5F9' },
  footer: { label: 'Footer', color: '#64748B', bg: '#F1F5F9' },
  page_number: { label: 'Page #', color: '#94A3B8', bg: '#F8FAFC' },
  footnote: { label: 'Footnote', color: '#64748B', bg: '#F8FAFC' },
  quote: { label: 'Quote', color: '#BE123C', bg: '#FFF1F2' },
  religious_quote: { label: 'Religious Quote', color: '#BE123C', bg: '#FFF1F2' },
  citation: { label: 'Citation', color: '#B45309', bg: '#FFFBEB' },
  list_item: { label: 'List item', color: '#0F172A', bg: '#F8FAFC' },
  paragraph: { label: 'Body', color: '#334155', bg: '#FFFFFF' }
};

// PHASE 5: pre-export structural validation. Read-only, non-destructive -
// it never deletes or alters any page; it only reports issues so the user
// can decide whether to proceed, fix a page, or export anyway.
const validateExportStructure = (sections) => {
  const warnings = [];

  // duplicate page ids (defensive - should not normally occur)
  const idCounts = {};
  sections.forEach(s => { idCounts[s.id] = (idCounts[s.id] || 0) + 1; });
  Object.entries(idCounts).forEach(([id, count]) => {
    if (count > 1) {
      warnings.push({ type: 'duplicate_page', pageId: id, message: `Page ${id} appears ${count} times in the document - this may produce duplicate exported sections.` });
    }
  });

  sections.forEach(s => {
    if (s.translationStatus !== 'done') {
      warnings.push({
        type: 'missing_translation',
        pageId: s.id,
        message: `Page ${s.id} has not been translated yet (status: ${s.translationStatus}) and will be skipped in the export.`
      });
      return;
    }
    const stripped = (s.translatedHtml || '').replace(/<[^>]*>/g, '').trim();
    if (!s.translatedHtml || !s.translatedHtml.trim()) {
      warnings.push({ type: 'empty_section', pageId: s.id, message: `Page ${s.id} is marked translated but has no content - it will export as an empty section.` });
    } else if (stripped.length < 3 && !/No Text Found/i.test(stripped)) {
      warnings.push({ type: 'malformed_section', pageId: s.id, message: `Page ${s.id}'s translated content looks malformed or unreadable.` });
    }
  });

  return warnings;
};

const App = () => {
  const [fileData, setFileData] = useState(null);
  const [parsedSections, setParsedSections] = useState([]);
  const [activeSectionId, setActiveSectionId] = useState(null);
  const [isParsing, setIsParsing] = useState(false);
  const [parseProgress, setParseProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);

  const [sourceLangMode, setSourceLangMode] = useState('auto');
  // targetLang: 'bn' (Bangla, valid for all 3 sources) or 'en' (English,
  // valid for Arabic/Urdu sources only - English -> English is meaningless).
  const [targetLang, setTargetLang] = useState('bn');
  const [isTranslatingAll, setIsTranslatingAll] = useState(false);
  const [progress, setProgress] = useState(0);

  // PHASE 3: document-level glossary/terminology memory (source -> Bangla).
  // Purely additive state; old sessions without it restore to [].
  const [glossary, setGlossary] = useState([]);
  // Ref mirror of `glossary` so in-flight async translation calls always read
  // the latest merged glossary without waiting on a React re-render, the same
  // pattern already used for rotationRef/pageImageCacheRef.
  const glossaryRef = useRef([]);

  const [apiKeys, setApiKeys] = useState(() => {
    try {
      const raw = sessionStorage.getItem('translator_api_keys') || localStorage.getItem('translator_api_keys');
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length === TOTAL_API_KEY_SLOTS) return parsed;
      }
    } catch (e) {}
    return Array(TOTAL_API_KEY_SLOTS).fill("");
  });
  const [rememberApiKey, setRememberApiKey] = useState(() => !!localStorage.getItem('translator_api_keys'));
  const [showSettings, setShowSettings] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // PHASE 4: review mode ('target' | 'compare' | 'overlay') - additive, defaults to 'target'
  // so existing behavior is unchanged unless the user explicitly switches mode.
  const [reviewMode, setReviewMode] = useState('target');
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  // PHASE 4: original page raster cache for Compare/Overlay modes, keyed by pageId.
  const [sourcePageImages, setSourcePageImages] = useState({});
  const [loadingSourceImage, setLoadingSourceImage] = useState(false);
  // PHASE 4: per-page diagnostics panel toggle (collapsed by default, non-intrusive).
  const [diagnosticsOpenFor, setDiagnosticsOpenFor] = useState({});

  // PHASE 5: pre-export warnings modal state. Purely additive; export still
  // works exactly as before when there are no warnings to show.
  const [exportWarnings, setExportWarnings] = useState([]);
  const [showExportWarningsModal, setShowExportWarningsModal] = useState(false);

  const pdfDocRef = useRef(null);
  const rotationRef = useRef({ keyIdx: 0, deadKeys: new Set() });
  const throttleRef = useRef(Promise.resolve());
  const pageImageCacheRef = useRef({});
  const lastRequestTimeRef = useRef((() => {
    try {
      const saved = Number(localStorage.getItem(LAST_REQUEST_TIME_STORAGE_KEY));
      return Number.isFinite(saved) ? saved : 0;
    } catch (e) {
      return 0;
    }
  })());
  const hasRestoredSession = useRef(false);
  const SESSION_STORAGE_KEY = 'translator_session_v2';
  const editorContainerRef = useRef(null); // PHASE 4: for image control injection

  useEffect(() => {
    try {
      const saved = localStorage.getItem(SESSION_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && Array.isArray(parsed.parsedSections) && parsed.parsedSections.length > 0) {
          const upgradedSections = parsed.parsedSections.map(s => ({
            ...s,
            previousTranslatedHtml: s.previousTranslatedHtml || "",
            translationError: s.translationError || null,
            retryCount: typeof s.retryCount === 'number' ? s.retryCount : 0,
            structure: s.structure || null
          }));
          setFileData(parsed.fileData || null);
          setParsedSections(upgradedSections);
          setActiveSectionId(parsed.activeSectionId ?? upgradedSections[0].id);
          setSourceLangMode(parsed.sourceLangMode || 'auto');
          setTargetLang(parsed.targetLang || 'bn');
          // PHASE 3: glossary restore - safe empty default for pre-Phase-3 sessions
          const restoredGlossary = Array.isArray(parsed.glossary) ? parsed.glossary : [];
          setGlossary(restoredGlossary);
          glossaryRef.current = restoredGlossary;
          setSuccessMsg("Restored your previous in-progress session.");
        }
      }
    } catch (e) {
      console.warn("Could not restore previous session:", e);
    } finally {
      hasRestoredSession.current = true;
    }
  }, []);

  useEffect(() => {
    if (!hasRestoredSession.current) return;
    try {
      if (parsedSections.length > 0) {
        // PHASE 3: glossary included in persisted session payload
        const payload = JSON.stringify({ fileData, parsedSections, activeSectionId, sourceLangMode, targetLang, glossary });
        localStorage.setItem(SESSION_STORAGE_KEY, payload);
      } else {
        localStorage.removeItem(SESSION_STORAGE_KEY);
      }
    } catch (e) {
      console.warn("Could not persist session (storage quota likely exceeded):", e);
    }
  }, [fileData, parsedSections, activeSectionId, sourceLangMode, targetLang, glossary]);

  // English target only makes sense for Arabic/Urdu sources (English -> English
  // is meaningless) - force back to 'auto' if this combination is selected.
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

        .translated-text-font {
          font-family: "Times New Roman", Times, serif !important;
        }

        .arabic-urdu-font {
          font-family: "Scheherazade New", "Noto Naskh Arabic", serif !important;
        }

        .english-font {
          font-family: "Plus Jakarta Sans", sans-serif !important;
        }
        
        .mirror-flow { 
          @apply leading-relaxed tracking-normal bg-white p-6 md:p-8 rounded-xl border border-slate-100 transition-all; 
        }

        /* PHASE 4: high-fidelity target "page" presentation - centered, bounded width,
           source-aspect-ratio aware, distinct rendering context per page. */
        .phase4-page-shell {
          @apply mx-auto bg-white rounded-lg border border-slate-200 shadow-lg;
          width: 100%;
          max-width: 880px;
          position: relative;
          isolation: isolate; /* independent stacking/rendering context per page */
        }
        .phase4-page-inner {
          padding: 2.25rem 2rem;
        }
        .phase4-workspace-bg {
          @apply bg-slate-100;
        }

        .phase4-diagram-controls {
          display: flex;
          gap: 6px;
          justify-content: center;
          margin-top: 8px;
        }
        .phase4-diagram-btn {
          font-size: 10px;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 8px;
          cursor: pointer;
          border: 1px solid transparent;
        }
        .phase4-diagram-btn.replace {
          background: #EEF2FF;
          color: #4338CA;
          border-color: #E0E7FF;
        }
        .phase4-diagram-btn.replace:hover { background: #E0E7FF; }
        .phase4-diagram-btn.remove {
          background: #FEF2F2;
          color: #B91C1C;
          border-color: #FEE2E2;
        }
        .phase4-diagram-btn.remove:hover { background: #FEE2E2; }

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

  // PHASE 5: sanitizeHtml hardened - explicit allowance for structured content
  // introduced in Phases 2-4 (table markup, local RTL spans, block-id markers,
  // phase4 asset index markers) so nothing legitimate is stripped on export,
  // while still using DOMPurify (not regex) as the primary sanitizer and never
  // widening it to unsafe tags/attributes (no <script>, no on*=, no iframes).
  const sanitizeHtml = useCallback((html) => {
    if (!html) return html;
    if (window.DOMPurify) {
      return window.DOMPurify.sanitize(html, {
        ADD_TAGS: ['table', 'thead', 'tbody', 'tr', 'td', 'th'],
        ADD_ATTR: ['class', 'dir', 'style', 'data-block-id', 'data-phase4-asset-index', 'colspan', 'rowspan']
      });
    }
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/ on[a-z]+="[^"]*"/gi, '')
      .replace(/ on[a-z]+='[^']*'/gi, '');
  }, []);

  // PHASE 1 FIX (issue 1.9): the old approach embedded a raw <input type="file">
  // directly inside the contentEditable translation editor's HTML. Clicking a
  // file input nested inside a contentEditable ancestor is unreliable across
  // browsers - contentEditable regions commonly intercept the click for text
  // caret placement instead of letting it reach the native file input, so the
  // "upload option" looked like it didn't exist even though the markup was
  // technically there. Initial upload now reuses the same reliable, already-
  // working pattern as "Replace Image" below: a temporary file input created
  // OUTSIDE the contentEditable tree (appended to document.body, clicked
  // programmatically, then removed) - see the post-render effect further
  // down, which wires an onClick on the placeholder itself to
  // handleReplaceUploadedImage for both the "not yet uploaded" and "replace
  // an existing upload" cases alike.

  // PHASE 4: keep a window-level mirror of parsedSections so the (module-scope)
  // file-select handler above can look up Phase 2 visual asset geometry for the
  // page/placeholder being replaced, without changing that handler's existing
  // detached-clone replacement approach (protected item #35).
  useEffect(() => {
    window.__phase4ParsedSectionsRef = parsedSections;
  }, [parsedSections]);

  const extractPageImageBase64 = async (pageId) => {
    if (pageImageCacheRef.current[pageId]) {
      return pageImageCacheRef.current[pageId];
    }
    if (!pdfDocRef.current) return null;
    try {
      const page = await pdfDocRef.current.getPage(pageId);
      // PHASE 4: raised render scale (2.0 -> 3.0) and JPEG quality (0.85 -> 0.94)
      // so small Arabic/Urdu diacritics, footnotes, seals, and logos stay legible.
      // Still cached exactly as before (item #58 protected) - no repeated re-render cost.
      const viewport = page.getViewport({ scale: 3.0 });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: ctx, viewport }).promise;
      const base64 = canvas.toDataURL('image/jpeg', 0.94).split(',')[1];
      pageImageCacheRef.current[pageId] = base64;
      return base64;
    } catch (e) {
      console.warn("Visual image extraction skipped for page", pageId, e);
      return null;
    }
  };

  // PHASE 4: fetch (and cache in component state) the full-page raster for
  // Compare/Overlay modes. Reuses the same cache/extraction function above -
  // no duplicate rendering pipeline.
  const ensureSourcePageImage = useCallback(async (pageId) => {
    if (!pageId || sourcePageImages[pageId]) return;
    setLoadingSourceImage(true);
    try {
      const base64 = await extractPageImageBase64(pageId);
      if (base64) {
        setSourcePageImages(prev => ({ ...prev, [pageId]: base64 }));
      }
    } finally {
      setLoadingSourceImage(false);
    }
  }, [sourcePageImages]);

  useEffect(() => {
    if ((reviewMode === 'compare' || reviewMode === 'overlay') && activeSectionId) {
      ensureSourcePageImage(activeSectionId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewMode, activeSectionId]);

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

    pageImageCacheRef.current = {};
    setSourcePageImages({}); // PHASE 4: reset cached source rasters for the new document
    // PHASE 3: a brand-new document starts a fresh glossary too, since
    // terminology consistency is scoped per document.
    setGlossary([]);
    glossaryRef.current = [];

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
          const baseViewport = page.getViewport({ scale: 1.0 });
          const pageWidth = baseViewport.width;
          const pageHeight = baseViewport.height;

          const isRtlPage = sourceLangMode === 'ar_ur';
          const positionedItems = textContent.items
            .filter(item => item.str !== undefined)
            .map((item, sourceIndex) => ({
              str: item.str,
              x: item.transform ? item.transform[4] : 0,
              y: item.transform ? item.transform[5] : 0,
              width: item.width || 0,
              height: item.height || (item.transform ? Math.hypot(item.transform[2], item.transform[3]) : 0),
              transform: item.transform || null,
              fontName: item.fontName || null,
              fontSize: item.transform ? Math.hypot(item.transform[2], item.transform[3]) : null,
              sourceIndex
            }));

          const lineTolerance = 3;
          const sortedForRawText = [...positionedItems].sort((a, b) => {
            if (Math.abs(a.y - b.y) > lineTolerance) return b.y - a.y;
            return isRtlPage ? b.x - a.x : a.x - b.x;
          });

          let pageRawText = sortedForRawText.map(item => item.str).join(" ");

          const lines = groupItemsIntoLines(positionedItems, lineTolerance);
          const avgFontSize = lines.reduce((sum, l) => sum + (l.fontSize || 0), 0) / (lines.length || 1);

          const columnInfo = detectColumns(lines, pageWidth);
          const orderedLines = buildColumnReadingOrder(lines, columnInfo, isRtlPage);
          if (columnInfo.columnCount > 1) {
            pageRawText = orderedLines.map(l => l.text).join(" ");
          }

          const blocks = classifyBlocks(orderedLines, i, pageHeight, avgFontSize);
          const tableCandidates = detectTableCandidates(orderedLines, pageWidth);
          const footnoteMeta = buildFootnoteMetadata(blocks);
          const continuation = detectLikelyContinuation(blocks);
          const visualAssets = await extractVisualAssetMetadata(page, i);

          const structure = {
            pageGeometry: {
              width: pageWidth,
              height: pageHeight,
              aspectRatio: pageHeight ? pageWidth / pageHeight : null,
              orientation: pageWidth >= pageHeight ? 'landscape' : 'portrait'
            },
            lines: orderedLines.map(l => ({
              lineIndex: l.lineIndex,
              x: l.x,
              y: l.y,
              width: l.width,
              height: l.height,
              fontSize: l.fontSize,
              text: l.text
            })),
            columns: columnInfo,
            blocks,
            tableCandidates,
            footnotes: footnoteMeta,
            continuation,
            visualAssets
          };

          sections.push({
            id: i,
            meta: { partName: `Page ${i}` },
            content: { rawText: pageRawText },
            translatedHtml: "",
            originalTranslatedHtml: "",
            previousTranslatedHtml: "",
            translationError: null,
            retryCount: 0,
            translationStatus: 'idle',
            isPdf: true,
            structure,
            // v7: Phase A's locked structural/visual JSON, and Phase B's
            // translated block texts - stored separately and persistently so
            // Retranslate (Phase B only) and Retry (whichever phase failed)
            // can reuse whichever phase already succeeded without redoing it.
            extraction: null,
            translatedBlocks: null
          });
        }
      }

      const withRepetition = computeRepeatedHeaderFooterMetadata(
        sections.map(s => ({ id: s.id, blocks: s.structure?.blocks || [] }))
      );
      withRepetition.forEach((rep, idx) => {
        if (sections[idx] && sections[idx].structure) {
          sections[idx].structure.repeatedHeaderFooter = rep.repeatedHeaderFooter;
        }
      });

      if (sections.length > 0) {
        console.info(
          `[PHASE 2] Parsed ${sections.length} page(s). Example structure for page 1:`,
          sections[0].structure
        );
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

  // callMistral keeps the exact same (Gemini-shaped) call signature and
  // return value as the old callGemini did - callers still pass a Gemini-style
  // `parts` array ([{text}, {inlineData:{mimeType,data}}]) and still read the
  // reply back out as data.candidates[0].content.parts[0].text. Internally it
  // now: (1) converts that into Mistral's message/content format, (2) sends
  // it to our own CORS proxy (never directly to api.mistral.ai - browsers
  // can't call that API cross-origin), and (3) reshapes Mistral's reply back
  // into the Gemini response shape before returning. This means NONE of the
  // downstream logic (glossary parsing, diagnostics, validation, etc.) had to
  // change - only this one function's internals did.
  // PHASE 2: callMistral now accepts an optional systemText as a second
  // argument. When provided, it's sent as a genuine system-role message,
  // with only the page-specific parts (page text, glossary/context hints,
  // and the image) in the user-role message. Models generally follow
  // system-level instructions more consistently than instructions buried
  // inside one long user turn - moving the STABLE rules (anti-substitution,
  // footnote handling, output format) to system and leaving only the
  // PAGE-SPECIFIC content in user is the goal here. Callers that don't pass
  // systemText keep working exactly as before (single user-role message).
  const callMistral = async (parts, systemText = null) => {
    const activeKeys = apiKeys.map(k => k.trim()).filter(Boolean);
    if (activeKeys.length === 0) {
      throw new Error("NO_API_KEY");
    }

    const acquireSlot = scheduleRequestSlot;
    const rot = rotationRef.current;

    if (rot.keyIdx >= activeKeys.length) rot.keyIdx = 0;

    const content = parts.map(p => {
      if (p && typeof p.text === 'string') {
        return { type: 'text', text: p.text };
      }
      if (p && p.inlineData) {
        return {
          type: 'image_url',
          image_url: { url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}` }
        };
      }
      return null;
    }).filter(Boolean);

    const messages = systemText
      ? [{ role: 'system', content: systemText }, { role: 'user', content }]
      : [{ role: 'user', content }];

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
            messages,
            // PHASE 0: low temperature - this is a mechanical mirroring/
            // translation task, not creative writing. A lower value reduces
            // drift into paraphrasing, wrong-sentence substitution, and
            // improvised footnote handling instead of following the strict
            // format the prompt asks for.
            temperature: 0.15,
            // PHASE 0: generous max_tokens (well under the account's 256k
            // combined input+output cap) so a dense page's full translated
            // HTML + footnotes + trailing glossary JSON is never silently
            // truncated mid-response - a cut-off response looks identical to
            // "the model dropped the footnote/paragraph", but has nothing to
            // do with model capability.
            max_tokens: 16000
          })
        }, 60000, acquireSlot);

        // Reshape Mistral's { choices: [{ message: { content } }] } reply
        // into the Gemini { candidates: [{ content: { parts: [{ text }] } }] }
        // shape every downstream caller already expects.
        const replyText = data?.choices?.[0]?.message?.content || "";
        return { candidates: [{ content: { parts: [{ text: replyText }] } }] };
      } catch (error) {
        const status = error.status;
        const isQuotaError = status === 429 || /rate.?limit|quota/i.test(error.message || "");
        const isKeyError = status === 400 || status === 401 || status === 403 || /invalid_api_key|unauthorized|permission|tier_not_allowed/i.test(error.message || "");

        if (isQuotaError) {
          if (error.retryAfterMs && error.retryAfterMs > 0) {
            const waitMs = Math.min(error.retryAfterMs, MAX_RETRY_AFTER_MS);
            await new Promise(resolve => setTimeout(resolve, waitMs));
          }
          // Rotate to next key; this key is NOT marked dead, it may recover
          // on a later cycle (e.g. next billing period / limit reset).
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

  // PHASE 3: translatePage now also accepts this page's structure,
  // the previous page's structure (for small cross-page context), and the
  // current glossary list, so the prompt can add block correspondence,
  // context, numeral, mixed-direction, and glossary-consistency guidance on
  // top of the existing (protected) anti-substitution prompt. All new
  // sections are optional and degrade to the original whole-page prompt
  // behavior when structure/glossary data isn't available.
  // v5 CALL 3 (CONDITIONAL): dedicated OCR pass, only when the page's
  // pdf.js-extracted text layer looks empty/broken. Most pages already have
  // a perfectly good text layer extracted for free during upload, so this
  // should rarely fire - it is NOT a fixed step every page goes through.
  // =============================================================================
  // v7 PHASE A - "EXTRACTION": the ONLY call that sees the image and makes ANY
  // structural/visual judgment (block type, alignment, size tier, color, header/
  // footer line breakdown, page number, footnote marker+text, image detection,
  // URL/icon detection, where inline emphasis/footnote-ref tokens belong).
  // Output is JSON, entirely in the ORIGINAL language - nothing is translated
  // here. Also absorbs the old conditional-OCR logic: since this call already
  // needs the image for every other judgment, a separate OCR call is no longer
  // needed - if the text layer looks broken, the same call just relies more on
  // the image for reading text too.
  // =============================================================================
  const runExtraction = async (pageText, pageId, isPdf, structure, prevPageStructure) => {
    let imagePayload = null;
    if (isPdf) {
      const base64Image = await extractPageImageBase64(pageId);
      if (base64Image) {
        imagePayload = { inlineData: { mimeType: "image/jpeg", data: base64Image } };
      }
    }

    const contextSection = buildContextPromptSection(structure, prevPageStructure);
    const estimatedParagraphCount = estimateBodyParagraphCount(structure);
    const textLooksBroken = !pageText || pageText.trim().length < 20;

    const extractionSystemPrompt = `
      You are analyzing ONE page's STRUCTURE and VISUAL APPEARANCE only. Do NOT translate anything - every "text" field you output stays in the page's ORIGINAL language (Arabic, Urdu, or English), exactly as printed. A separate step handles translation afterward, working entirely from what you output here - so be thorough and precise, since anything you miss or mis-tag will not get a second chance to be caught downstream.

      ${textLooksBroken ? 'The extracted text layer for this page looks empty or broken - rely ENTIRELY on the page image to read the text accurately (this is effectively an OCR task for this page).' : "You're given an extracted text layer below as a starting point - cross-reference it against the image, correcting anything that looks wrong."}
      ${contextSection ? `\n      DOCUMENT CONTEXT (for judging whether this page's first paragraph continues from the previous page, etc - does not change what text belongs to THIS page):\n${contextSection}\n` : ''}

      Output ONLY a single JSON object (no markdown fences, no commentary before or after), in exactly this shape:
      {
        "pageNumber": { "text": "...", "align": "center" } or null if none visible,
        "header": { "lines": [ { "id": "h1", "text": "...", "align": "left", "sizeTier": "small" } ] } or { "lines": [] } if none,
        "footer": { "lines": [ { "id": "f1", "text": "...", "align": "center", "sizeTier": "small" } ] } or { "lines": [] } if none,
        "blocks": [ ... see block shape below ... ],
        "paragraphCount": <number of distinct body paragraphs you identified>
      }

      EACH block in "blocks" (in top-to-bottom reading order) has this shape:
      { "id": "b1", "type": "...", "text": "...", "align": "left|center|right|justify", "sizeTier": "large_heading|medium_heading|subheading|body|small", "color": "#RRGGBB or null", "marker": "1 (footnote blocks only)", "preserve": true/false }

      TYPE options and when to use each:
      - "heading" / "subheading": a title or section header, larger/bolder than body text
      - "paragraph": ordinary body text
      - "quote" / "citation" / "list_item": same body styling, but tagged for reference
      - "footnote": a footnote/reference entry from the bottom of the page. MUST include a "marker" field with the exact reference number/symbol as printed (e.g. "1", "٢", "*") - do not renumber, do not convert its digit script.
      - "image": a REAL visual element (photograph, diagram, chart, map, stamp, seal, logo, signature - NOT dense calligraphic text, NOT a decorative divider line, NOT stylized-but-still-readable heading text; if you can read it as a sentence, it's text, not an image). No "text" field needed for images, just its position in the block order.
      - "url": a literal web address / domain name. Set "text" to the exact URL and "preserve": true.
      - "icon": a small inline symbol/glyph that isn't a full image and isn't translatable text. Set "preserve": true.

      ALIGNMENT: observe each block's actual alignment in the image (left/center/right/justify) rather than assuming.
      SIZE TIER: judge each block's size RELATIVE to the page's own normal body text: "large_heading" (clearly the biggest element, e.g. a chapter title), "medium_heading" (clearly bigger than body but not the biggest), "subheading" (only slightly bigger/bolder than body), "body" (normal running text size), "small" (visibly smaller, e.g. footnotes/captions/fine print).
      COLOR: only set a non-null color if the block is ACTUALLY colored in the source (e.g. a colored heading) - most blocks should have color: null and inherit the default styling downstream.
      ${estimatedParagraphCount ? `A local structural scan estimates approximately ${estimatedParagraphCount} body paragraph(s) on this page - use this as a sanity check on your own paragraph segmentation, but trust your own visual reading if it clearly disagrees.` : ''}

      --- INLINE TOKENS - MARK, DO NOT FORMAT ---
      Two things need to be marked INSIDE a block's "text" field using plain-text tokens (NOT real HTML - a later step converts these, you only place them):
      1. Where a footnote reference marker appears INSIDE body text (e.g. mid-sentence, referring to a footnote), insert the literal token ⟦FN:marker⟧ at that exact position, using the same marker value as the corresponding footnote block. Example: "...as reported¹" becomes "...as reported⟦FN:1⟧".
      2. Where a word or phrase is visibly emphasized/highlighted in the source (bold+colored key terms, Quranic verse excerpts, etc), wrap just that span with ⟦HL⟧...⟦/HL⟧. Example: "the Prophet ﷺ said" with "said" emphasized becomes "the Prophet ﷺ ⟦HL⟧said⟧/HL⟧" (note: literally type ⟦/HL⟧ to close).
      Do not use any other markup inside "text" fields - no HTML tags, only these two token types where they genuinely apply.

      --- BOOK/WORK TITLES - MARK FOR PRESERVATION, DO NOT TRANSLATE THE TITLE ITSELF ---
      If a footnote or citation names a specific book/published work, keep that title in the "text" field in its ORIGINAL script exactly as printed (a later step will preserve it, not translate it) - but you may still include surrounding words normally, since only the title portion will be protected from translation. Simplest approach: set "preserve": true on the whole footnote/citation block ONLY if it is entirely just a title/URL with nothing else to translate; otherwise leave preserve false and the title will still read correctly in the original text field.
    `;

    const extractionUserPrompt = `
      PAGE TEXT LAYER (starting point - verify/correct against the image):
      """
      ${pageText || '(empty - rely on image)'}
      """
      Analyze this page now and output the JSON object described in the system message.
    `;

    try {
      const data = await callMistral([
        { text: extractionUserPrompt },
        ...(imagePayload ? [imagePayload] : [])
      ], extractionSystemPrompt);

      const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
      const parsed = parseJsonFromModelResponse(raw);
      if (!parsed || !Array.isArray(parsed.blocks)) return null;
      return parsed;
    } catch (e) {
      console.error("Extraction (Phase A) call failed:", e);
      throw e; // let the caller distinguish this from a Phase B failure
    }
  };

  // =============================================================================
  // v7 PHASE B - "TRANSLATION": receives ONLY {id, text} pairs for translatable
  // (non-preserve) blocks - no image, no formatting decisions. Its ONLY job is
  // swapping original-language words for target-language words, leaving the two
  // inline token types (⟦FN:..⟧ / ⟦HL⟧..⟦/HL⟧) exactly as given, untouched and
  // unmoved - it does not decide WHERE emphasis or footnote references go, only
  // translates the words around and inside them.
  // =============================================================================
  const runTranslationPhase = async (extraction, targetCode, glossaryList, structure) => {
    const targetName = TARGET_LABELS[targetCode];
    const glossaryExcerpt = buildGlossaryPromptExcerpt(glossaryList);
    const repeatedMetadataHints = buildRepeatedMetadataPromptSection(structure, glossaryList);

    const translatable = [];
    (extraction.header?.lines || []).forEach(l => translatable.push({ id: `header:${l.id}`, text: l.text }));
    (extraction.footer?.lines || []).forEach(l => translatable.push({ id: `footer:${l.id}`, text: l.text }));
    (extraction.blocks || []).forEach(b => {
      if (!b.preserve && b.type !== 'image' && typeof b.text === 'string') {
        translatable.push({ id: b.id, text: b.text });
      }
    });

    if (translatable.length === 0) {
      return { translationMap: {}, glossarySuggestions: [] };
    }

    const translationSystemPrompt = `
      You translate ONLY the words given to you into ${targetName}. Every structural/visual decision (block type, alignment, size, position) has ALREADY been made by a separate step - you never see or need to make those decisions. Your entire job is: for each {id, text} entry given, produce {id, translatedText}.

      ============================================================
      RULE #1 - NO CONTENT SUBSTITUTION
      ============================================================
      Translate exactly the words given in each entry's "text" - never blend in wording from a different entry, even if they're topically related (e.g. a footnote entry and a nearby paragraph entry). Each entry is translated strictly from its own given text.

      --- PRESERVE THESE TOKENS EXACTLY, UNTRANSLATED, UNMOVED ---
      Some entries' text contains literal tokens: ⟦FN:marker⟧ (a footnote reference point) or ⟦HL⟧...⟦/HL⟧ (an emphasis span wrapper). These are NOT words to translate - copy them into your output EXACTLY as given, in the same relative position within the sentence, translating only the actual words around/inside them. Do not add, remove, or move these tokens.

      --- BOOK/WORK TITLES - DO NOT TRANSLATE ---
      If a specific book or published work's title appears in the text (e.g. "Sahih Al-Bukhari"), keep that title exactly as given in its original script/spelling - do not translate it, even though you translate the surrounding words normally.
${buildNumeralHandlingInstructions(targetCode)}
      ${glossaryExcerpt ? `--- APPROVED TERMINOLOGY GLOSSARY (reuse for consistency, unless context clearly requires otherwise) ---\n${glossaryExcerpt}\n` : ''}
      ${repeatedMetadataHints ? `--- REPEATED HEADER/FOOTER WORDING - KEEP CONSISTENT ---\n${repeatedMetadataHints}\n` : ''}

      Output ONLY a single JSON object (no markdown fences, no commentary), in exactly this shape:
      {
        "translations": [ { "id": "b1", "translatedText": "..." }, ... one entry per input id, same ids, same order ... ],
        "glossarySuggestions": [ { "source": "...", "translated": "..." }, ... at most 8 important recurring terms from this page, or [] if none ... ]
      }
    `;

    const translationUserPrompt = `
      ENTRIES TO TRANSLATE:
      """
${translatable.map(t => `      ID: ${t.id}\n      TEXT: ${t.text}`).join('\n\n')}
      """
      Translate every entry into ${targetName} and output the JSON object described in the system message.
    `;

    try {
      const data = await callMistral([{ text: translationUserPrompt }], translationSystemPrompt);
      const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
      const parsed = parseJsonFromModelResponse(raw);
      if (!parsed || !Array.isArray(parsed.translations)) return null;

      const translationMap = {};
      parsed.translations.forEach(t => {
        if (t && t.id) translationMap[t.id] = t.translatedText || '';
      });
      const glossarySuggestions = Array.isArray(parsed.glossarySuggestions) ? parsed.glossarySuggestions : [];
      return { translationMap, glossarySuggestions };
    } catch (e) {
      console.error("Translation (Phase B) call failed:", e);
      throw e; // let the caller distinguish this from a Phase A failure
    }
  };

  // v7 orchestrator: runs Phase A, then Phase B, then assembles via pure code
  // (assemblePage - zero AI judgment). Returns which phase (if any) failed, so
  // the caller can set a precise 'failed-phase-a' / 'failed-phase-b' status and
  // persist whichever phase DID succeed - Retry only needs to redo the phase
  // that actually failed, Retranslate only ever redoes Phase B.
  const translatePage = async (pageText, pageId, isPdf, structure = null, prevPageStructure = null, glossaryList = []) => {
    let extraction = null;
    try {
      extraction = await runExtraction(pageText, pageId, isPdf, structure, prevPageStructure);
    } catch (e) {
      return { phase: 'A', error: e, extraction: null, translationMap: null, html: '', glossarySuggestions: [] };
    }
    if (!extraction) {
      return { phase: 'A', error: new Error('Extraction returned invalid/unparseable JSON.'), extraction: null, translationMap: null, html: '', glossarySuggestions: [] };
    }

    let translationResult = null;
    try {
      translationResult = await runTranslationPhase(extraction, targetLang, glossaryList, structure);
    } catch (e) {
      return { phase: 'B', error: e, extraction, translationMap: null, html: '', glossarySuggestions: [] };
    }
    if (!translationResult) {
      return { phase: 'B', error: new Error('Translation returned invalid/unparseable JSON.'), extraction, translationMap: null, html: '', glossarySuggestions: [] };
    }

    const html = assemblePage(extraction, translationResult.translationMap, targetLang);
    return { phase: null, error: null, extraction, translationMap: translationResult.translationMap, html, glossarySuggestions: translationResult.glossarySuggestions };
  };

  // Verification & auto-correction pass.
  // v7 VERIFY - audits Phase A's extraction JSON only (never touches Phase B's
  // translated text). Same completeness-first checklist philosophy as before,
  // but the audit target is now the structural JSON, not assembled HTML -
  // consistent with "Verify only for Phase A" from the plan. After a
  // correction, the caller re-runs Phase B fresh against the corrected JSON
  // (simpler and safer than trying to diff and re-translate only the changed
  // blocks, at the cost of Phase B re-running in full on every Verify).
  const verifyExtraction = async (pageText, pageId, isPdf, existingExtraction, structure = null) => {
    let imagePayload = null;
    if (isPdf) {
      const base64Image = await extractPageImageBase64(pageId);
      if (base64Image) {
        imagePayload = { inlineData: { mimeType: "image/jpeg", data: base64Image } };
      }
    }

    const verifyBlockMapSection = buildBlockMapPromptSection(structure);

    const verifySystemPrompt = `
      YOU ARE A PROOFREADER/AUDITOR for a page-extraction JSON, not a translator. Your job is to CHECK an existing structural extraction (block types, text, alignment, size, color, header/footer lines, footnotes, images) against the original source page, and fix errors you find. Thoroughness matters more than brevity - take as much space as needed, and do not hesitate to add missing blocks entirely if you find real gaps.

      Check specifically, IN THIS PRIORITY ORDER:
      1. COMPLETENESS (CHECK FIRST, MOST CAREFULLY): compare the source page (text and image) against the JSON's blocks, from the very top to the very bottom, INCLUDING the last 2-3 paragraphs/lines before the page ends - that's exactly where content most often gets silently dropped. Is there any sentence, paragraph, heading, signature line, or footnote present in the source with NO corresponding block in the JSON? If so, ADD the missing block(s) now, in the correct position.
      ${verifyBlockMapSection ? `Reference - local structural scan's own block map, as a cross-check:\n${verifyBlockMapSection}\n` : ''}
      2. FOOTNOTE MARKERS: does every footnote block have the correct "marker" value exactly as printed in the source (not renumbered, not digit-converted)? Does every inline ⟦FN:marker⟧ token in body text have a matching footnote block with that same marker?
      3. TYPE ACCURACY: is any block mis-typed - e.g. a real image marked as a paragraph, or dense decorative text incorrectly marked as an "image" block?
      4. ALIGNMENT & SIZE: does each block's "align" and "sizeTier" roughly match what's actually observed in the source image?
      5. INLINE TOKENS: are ⟦HL⟧...⟦/HL⟧ emphasis spans correctly placed around genuinely emphasized/highlighted source text, not missing or misplaced?
      6. HEADER/FOOTER/PAGE NUMBER: are all header lines, footer lines, and the page number captured, each with correct text/alignment?

      OUTPUT RULES:
      - If you find NO issues after checking all six points, respond with EXACTLY this sentinel text and nothing else: NO_CORRECTIONS_NEEDED
      - If you find ANY issue, respond with the FULL corrected JSON object, in the exact same shape as the input JSON (pageNumber, header, footer, blocks, paragraphCount), with the errors fixed and any missing blocks added. Preserve blocks that were already correct unchanged.
      - Output ONLY the sentinel text alone, or raw JSON alone (no markdown fences) - never both, never any other commentary.
    `;

    const verifyUserPrompt = `
      ORIGINAL SOURCE PAGE TEXT (also see attached page image if provided):
      """
      ${pageText}
      """

      EXISTING EXTRACTION JSON TO AUDIT:
      """
      ${JSON.stringify(existingExtraction)}
      """

      Audit this extraction against the source following the checklist in the system message.
    `;

    try {
      const data = await callMistral([
        { text: verifyUserPrompt },
        ...(imagePayload ? [imagePayload] : [])
      ], verifySystemPrompt);

      const rawText = (data.candidates?.[0]?.content?.parts?.[0]?.text || "").trim();

      if (!rawText || /^NO_CORRECTIONS_NEEDED$/i.test(rawText)) {
        return existingExtraction; // no changes needed
      }

      const corrected = parseJsonFromModelResponse(rawText);
      return (corrected && Array.isArray(corrected.blocks)) ? corrected : existingExtraction;
    } catch (e) {
      console.error("Extraction verification error:", e);
      return existingExtraction; // never lose the existing good extraction on a failed audit call
    }
  };

  // PHASE 3: batch loop now carries each page's structure (for block map /
  // context) and looks up the previous page's structure by array position,
  // and merges any glossary suggestions returned per page into glossaryRef /
  // glossary state so later pages in the same batch benefit immediately.
  const startSequentialAnalysis = async () => {
    if (isTranslatingAll) return;
    setIsTranslatingAll(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const pageQueue = parsedSections.map(p => ({
      id: p.id,
      rawText: p.content.rawText,
      isPdf: p.isPdf,
      status: p.translationStatus,
      structure: p.structure || null
    }));
    const totalPages = pageQueue.length;

    for (let i = 0; i < totalPages; i++) {
      const { id: pageId, rawText, isPdf, status, structure } = pageQueue[i];
      const prevPageStructure = i > 0 ? pageQueue[i - 1].structure : null;

      if (status !== 'done') {
        setParsedSections(prev => prev.map(s => (
          s.id === pageId ? { ...s, translationStatus: 'loading', translationError: null } : s
        )));

        let translatedHtmlResult = null;
        let extractionResult = null;
        let translatedBlocksResult = null;
        let failureReason = null;
        let failedPhase = null;

        try {
          const result = await translatePage(rawText, pageId, isPdf, structure, prevPageStructure, glossaryRef.current);
          extractionResult = result.extraction; // may be non-null even if Phase B failed - keep it either way

          if (result.phase) {
            // A phase genuinely failed to produce usable output (API error or
            // unparseable JSON) - distinct from "produced output, but it
            // looked wrong", which validateTranslationResult catches below.
            failedPhase = result.phase;
            failureReason = result.error?.message || `Phase ${result.phase} failed.`;
          } else {
            const validation = validateTranslationResult(result.html, rawText);
            if (validation.valid) {
              translatedHtmlResult = result.html;
              translatedBlocksResult = result.translationMap;
              const merged = mergeGlossaryEntries(glossaryRef.current, result.glossarySuggestions);
              glossaryRef.current = merged;
              setGlossary(merged);
            } else {
              failureReason = validation.reason;
            }
          }
        } catch (e) {
          console.error("Translation error:", e);
          if (e.message === 'NO_API_KEY') {
            failureReason = "Please enter at least one Mistral API key in Settings.";
          } else if (e.message === 'ALL_KEYS_EXHAUSTED') {
            failureReason = "All Mistral API keys have reached their limit or are invalid. Please add more keys in Settings, or wait for a key's limit to reset.";
          } else {
            failureReason = `Translation failed: ${e.message}`;
          }
        }

        if (failureReason) setErrorMsg(failureReason);

        setParsedSections(prev => prev.map(s => {
          if (s.id !== pageId) return s;
          if (translatedHtmlResult) {
            return {
              ...s,
              extraction: extractionResult,
              translatedBlocks: translatedBlocksResult,
              translatedHtml: translatedHtmlResult,
              originalTranslatedHtml: s.originalTranslatedHtml || translatedHtmlResult,
              translationStatus: 'done',
              translationError: null
            };
          }
          return {
            ...s,
            extraction: extractionResult || s.extraction, // keep Phase A's result even if Phase B failed
            translationStatus: failedPhase === 'A' ? 'failed-phase-a' : failedPhase === 'B' ? 'failed-phase-b' : 'error',
            translationError: failureReason
          };
        }));
      }

      setProgress(Math.round(((i + 1) / totalPages) * 100));
    }

    setIsTranslatingAll(false);
    setSuccessMsg("Batch translation complete!");
  };

  // v7 VERIFY - Phase A only. Audits/corrects the extraction JSON, then
  // re-runs Phase B fresh against the corrected structure (Phase B is cheap
  // and this avoids the complexity/risk of trying to only re-translate the
  // specific blocks that changed).
  const handleVerifyExtraction = async (pageId) => {
    const page = parsedSections.find(s => s.id === pageId);
    if (!page || !page.extraction) return;

    setErrorMsg(null);
    setSuccessMsg(null);
    setParsedSections(prev => prev.map(s => (s.id === pageId ? { ...s, translationStatus: 'verifying' } : s)));

    try {
      const correctedExtraction = await verifyExtraction(page.content.rawText, page.id, page.isPdf, page.extraction, page.structure);
      const translationResult = await runTranslationPhase(correctedExtraction, targetLang, glossaryRef.current, page.structure);
      const finalHtml = assemblePage(correctedExtraction, translationResult?.translationMap || {}, targetLang);

      if (translationResult?.glossarySuggestions?.length) {
        const merged = mergeGlossaryEntries(glossaryRef.current, translationResult.glossarySuggestions);
        glossaryRef.current = merged;
        setGlossary(merged);
      }

      setParsedSections(prev => prev.map(s => (
        s.id === pageId
          ? {
              ...s,
              extraction: correctedExtraction,
              translatedBlocks: translationResult?.translationMap || s.translatedBlocks,
              translatedHtml: finalHtml,
              originalTranslatedHtml: s.originalTranslatedHtml || finalHtml,
              translationStatus: 'done'
            }
          : s
      )));
      setSuccessMsg(`Section ${pageId} extraction verified successfully!`);
    } catch (e) {
      console.error("Extraction verification failed:", e);
      setErrorMsg(`Verify failed: ${e.message}`);
      setParsedSections(prev => prev.map(s => (s.id === pageId ? { ...s, translationStatus: 'done' } : s)));
    }
  };

  // v7 RETRANSLATE - Phase B only. Reuses the already-stored Phase A
  // extraction unchanged - no image, no re-extraction, much cheaper/faster
  // than a full page redo. Useful specifically when the STRUCTURE was fine
  // but the WORDS came out wrong.
  const handleRetranslate = async (pageId) => {
    const page = parsedSections.find(s => s.id === pageId);
    if (!page || !page.extraction) return;

    setErrorMsg(null);
    setSuccessMsg(null);
    setParsedSections(prev => prev.map(s => (s.id === pageId ? { ...s, translationStatus: 'retranslating' } : s)));

    try {
      const translationResult = await runTranslationPhase(page.extraction, targetLang, glossaryRef.current, page.structure);
      if (!translationResult) throw new Error("Translation returned invalid/unparseable JSON.");

      const finalHtml = assemblePage(page.extraction, translationResult.translationMap, targetLang);
      const merged = mergeGlossaryEntries(glossaryRef.current, translationResult.glossarySuggestions);
      glossaryRef.current = merged;
      setGlossary(merged);

      setParsedSections(prev => prev.map(s => (
        s.id === pageId
          ? {
              ...s,
              previousTranslatedHtml: s.translatedHtml || s.previousTranslatedHtml,
              translatedBlocks: translationResult.translationMap,
              translatedHtml: finalHtml,
              originalTranslatedHtml: s.originalTranslatedHtml || finalHtml,
              translationStatus: 'done',
              translationError: null
            }
          : s
      )));
      setSuccessMsg(`Section ${pageId} retranslated successfully!`);
    } catch (e) {
      console.error("Retranslate failed:", e);
      setErrorMsg(`Retranslate failed: ${e.message}`);
      setParsedSections(prev => prev.map(s => (s.id === pageId ? { ...s, translationStatus: 'done' } : s)));
    }
  };

  // v7 RETRY - phase-aware: only redoes whichever phase actually failed.
  // If Phase A never succeeded (no stored extraction), retries BOTH phases
  // via the full translatePage orchestrator. If Phase A already succeeded
  // but Phase B failed, retries ONLY Phase B against the existing extraction
  // - cheaper and faster, and never re-does work that already succeeded.
  const handleRetryPage = async (pageId) => {
    const page = parsedSections.find(s => s.id === pageId);
    if (!page) return;
    const prevPage = parsedSections.find(s => s.id === pageId - 1);

    setErrorMsg(null);
    setSuccessMsg(null);
    setParsedSections(prev => prev.map(s => (
      s.id === pageId ? { ...s, translationStatus: 'loading', translationError: null } : s
    )));

    // Phase A already has a usable extraction stored - only Phase B needs retrying.
    if (page.extraction && page.translationStatus === 'failed-phase-b') {
      try {
        const translationResult = await runTranslationPhase(page.extraction, targetLang, glossaryRef.current, page.structure);
        if (!translationResult) throw new Error("Translation returned invalid/unparseable JSON.");
        const finalHtml = assemblePage(page.extraction, translationResult.translationMap, targetLang);
        const merged = mergeGlossaryEntries(glossaryRef.current, translationResult.glossarySuggestions);
        glossaryRef.current = merged;
        setGlossary(merged);

        setParsedSections(prev => prev.map(s => (
          s.id === pageId
            ? {
                ...s,
                translatedBlocks: translationResult.translationMap,
                translatedHtml: finalHtml,
                originalTranslatedHtml: s.originalTranslatedHtml || finalHtml,
                translationStatus: 'done',
                translationError: null,
                retryCount: (s.retryCount || 0) + 1
              }
            : s
        )));
        setSuccessMsg(`Section ${pageId} retried successfully (Phase B only)!`);
      } catch (e) {
        console.error("Retry (Phase B) failed:", e);
        const msg = `Retry failed: ${e.message}`;
        setErrorMsg(msg);
        setParsedSections(prev => prev.map(s => (
          s.id === pageId ? { ...s, translationStatus: 'failed-phase-b', translationError: msg, retryCount: (s.retryCount || 0) + 1 } : s
        )));
      }
      return;
    }

    // Otherwise (Phase A never succeeded, or a non-phase-specific error like
    // NO_API_KEY) - full retry of both phases via the normal orchestrator.
    let translatedHtmlResult = null;
    let extractionResult = null;
    let translatedBlocksResult = null;
    let failureReason = null;
    let failedPhase = null;
    try {
      const result = await translatePage(
        page.content.rawText,
        page.id,
        page.isPdf,
        page.structure || null,
        prevPage?.structure || null,
        glossaryRef.current
      );
      extractionResult = result.extraction;

      if (result.phase) {
        failedPhase = result.phase;
        failureReason = result.error?.message || `Phase ${result.phase} failed.`;
      } else {
        const validation = validateTranslationResult(result.html, page.content.rawText);
        if (validation.valid) {
          translatedHtmlResult = result.html;
          translatedBlocksResult = result.translationMap;
          const merged = mergeGlossaryEntries(glossaryRef.current, result.glossarySuggestions);
          glossaryRef.current = merged;
          setGlossary(merged);
        } else {
          failureReason = validation.reason;
        }
      }
    } catch (e) {
      console.error("Retry translation error:", e);
      if (e.message === 'NO_API_KEY') {
        failureReason = "Please enter at least one Mistral API key in Settings.";
      } else if (e.message === 'ALL_KEYS_EXHAUSTED') {
        failureReason = "All Mistral API keys have reached their limit or are invalid. Please add more keys in Settings, or wait for a key's limit to reset.";
      } else {
        failureReason = `Retry failed: ${e.message}`;
      }
    }

    if (failureReason) setErrorMsg(failureReason);

    setParsedSections(prev => prev.map(s => {
      if (s.id !== pageId) return s;
      if (translatedHtmlResult) {
        return {
          ...s,
          previousTranslatedHtml: s.translatedHtml || s.previousTranslatedHtml,
          extraction: extractionResult,
          translatedBlocks: translatedBlocksResult,
          translatedHtml: translatedHtmlResult,
          originalTranslatedHtml: s.originalTranslatedHtml || translatedHtmlResult,
          translationStatus: 'done',
          translationError: null,
          retryCount: (s.retryCount || 0) + 1
        };
      }
      return {
        ...s,
        extraction: extractionResult || s.extraction,
        translationStatus: failedPhase === 'A' ? 'failed-phase-a' : failedPhase === 'B' ? 'failed-phase-b' : 'error',
        translationError: failureReason,
        retryCount: (s.retryCount || 0) + 1
      };
    }));

    if (translatedHtmlResult) {
      setSuccessMsg(`Section ${pageId} retried successfully!`);
    }
  };

  const handleResetTranslation = (pageId) => {
    setParsedSections(prev => prev.map(s => {
      if (s.id === pageId && s.originalTranslatedHtml) {
        return {
          ...s,
          previousTranslatedHtml: s.translatedHtml || s.previousTranslatedHtml,
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
    setParsedSections(prev => {
      return prev.map(s => s.id === pageId ? { ...s, translatedHtml: updatedHtml } : s);
    });
  };

  // PHASE 4: "Remove Image" - reverts a specific uploaded placeholder back to
  // the original upload-prompt placeholder markup, without touching any other
  // placeholder on this page or any other page's images (protected item #58).
  const handleRemoveUploadedImage = useCallback((pageId, placeholderIndex) => {
    setParsedSections(prev => prev.map(s => {
      if (s.id !== pageId || !s.translatedHtml) return s;
      const wrapper = document.createElement('div');
      wrapper.innerHTML = s.translatedHtml;
      const placeholders = Array.from(wrapper.querySelectorAll('.diagram-placeholder'));
      const target = placeholders[placeholderIndex];
      if (!target) return s;
      target.style.border = '2px dashed #CBD5E1';
      target.style.background = '';
      target.style.padding = '20px';
      target.innerHTML = '<p style="font-size: 14px; color: #64748B; margin-bottom: 8px;">📤 Image Detected. Click here to upload replacement.</p>';
      return { ...s, translatedHtml: wrapper.innerHTML };
    }));
    setSuccessMsg('Image removed. You can upload a new one for this placeholder.');
  }, []);

  // PHASE 4: "Replace Image" - opens a fresh file picker for a specific
  // already-filled placeholder and swaps only that image, preserving its
  // position/identity (reuses the same detached-clone + diagramUploaded
  // event flow as the original upload path - protected items #33-#36).
  const handleReplaceUploadedImage = useCallback((pageId, placeholderIndex) => {
    const tempInput = document.createElement('input');
    tempInput.type = 'file';
    tempInput.accept = 'image/*';
    tempInput.style.display = 'none';
    document.body.appendChild(tempInput);

    tempInput.addEventListener('change', () => {
      const file = tempInput.files[0];
      document.body.removeChild(tempInput);
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target.result;
        setParsedSections(prev => prev.map(s => {
          if (s.id !== pageId || !s.translatedHtml) return s;
          const wrapper = document.createElement('div');
          wrapper.innerHTML = s.translatedHtml;
          const placeholders = Array.from(wrapper.querySelectorAll('.diagram-placeholder'));
          const target = placeholders[placeholderIndex];
          if (!target) return s;

          const assetMeta = s.structure?.visualAssets?.[placeholderIndex] || null;
          const hasKnownSize = assetMeta && assetMeta.width && assetMeta.height;
          const imgStyle = hasKnownSize
            ? `max-width: 100%; width: ${Math.min(assetMeta.width, 700)}px; height: auto; aspect-ratio: ${assetMeta.width} / ${assetMeta.height}; border-radius: 8px; display: block; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);`
            : `max-width: 100%; max-height: 400px; border-radius: 8px; display: block; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);`;

          target.innerHTML = `<img src="${base64}" data-phase4-asset-index="${placeholderIndex}" style="${imgStyle}" alt="Uploaded Diagram" />`;
          target.style.border = 'none';
          target.style.background = 'transparent';
          target.style.padding = '0';
          return { ...s, translatedHtml: wrapper.innerHTML };
        }));
        setSuccessMsg('Image replaced successfully.');
      };
      reader.readAsDataURL(file);
    });

    tempInput.click();
  }, []);

  // PHASE 4: after the editable content renders, inject "Replace"/"Remove"
  // controls under each already-filled diagram placeholder for the active
  // page only. This is purely a DOM-presentation pass on top of the existing
  // rendered HTML - it never mutates parsedSections/translatedHtml itself, so the
  // underlying translation content and export output are untouched.
  useEffect(() => {
    const container = editorContainerRef.current;
    if (!container) return;
    const editorDiv = container.querySelector('[id^="translation-editor-"]');
    if (!editorDiv) return;

    const placeholders = Array.from(editorDiv.querySelectorAll('.diagram-placeholder'));
    placeholders.forEach((placeholder, idx) => {
      const hasImage = !!placeholder.querySelector('img');
      let controls = placeholder.parentElement && placeholder.parentElement.classList.contains('phase4-diagram-controls-wrapper')
        ? placeholder.parentElement.querySelector('.phase4-diagram-controls')
        : null;

      if (!hasImage) {
        // PHASE 1 FIX (issue 1.9): clicking the placeholder now opens a file
        // picker via handleReplaceUploadedImage's outside-contentEditable
        // temp-input pattern (the same reliable mechanism the "Replace
        // Image" button already used successfully) - not an embedded
        // <input type="file"> that contentEditable was swallowing clicks on.
        placeholder.style.cursor = 'pointer';
        placeholder.onclick = (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          handleReplaceUploadedImage(activeSectionId, idx);
        };
        if (controls) controls.remove();
        return;
      }

      if (!controls) {
        controls = document.createElement('div');
        controls.className = 'phase4-diagram-controls';
        const replaceBtn = document.createElement('button');
        replaceBtn.type = 'button';
        replaceBtn.className = 'phase4-diagram-btn replace';
        replaceBtn.textContent = 'Replace Image';
        replaceBtn.onclick = (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          handleReplaceUploadedImage(activeSectionId, idx);
        };
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'phase4-diagram-btn remove';
        removeBtn.textContent = 'Remove Image';
        removeBtn.onclick = (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          handleRemoveUploadedImage(activeSectionId, idx);
        };
        controls.appendChild(replaceBtn);
        controls.appendChild(removeBtn);
        placeholder.insertAdjacentElement('afterend', controls);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSectionId, parsedSections, reviewMode, handleReplaceUploadedImage, handleRemoveUploadedImage]);

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

    sessionStorage.setItem('translator_api_keys', JSON.stringify(keys));

    if (rememberApiKey) {
      localStorage.setItem('translator_api_keys', JSON.stringify(keys));
    } else {
      localStorage.removeItem('translator_api_keys');
    }

    setApiKeys(keys);
    rotationRef.current = { keyIdx: 0, deadKeys: new Set() };
    setShowSettings(false);
    setSuccessMsg("Settings saved successfully.");
  };

  // PHASE 5: performExport contains the actual export-generation logic
  // (hardened per Phase 5 requirements). It is only reached after validation
  // (see exportHtml below) has either found no warnings, or the user chose
  // to proceed anyway from the warnings modal.
  const performExport = async () => {
    if (isExporting) return;
    setIsExporting(true);
    setErrorMsg(null);

    const finishedTranslations = parsedSections.filter(s => s.translationStatus === 'done' && s.translatedHtml && s.translatedHtml.trim());
    if (finishedTranslations.length === 0) {
      setErrorMsg("No translated pages found. Please process at least one section first.");
      setIsExporting(false);
      return;
    }

    try {
      // PHASE 5: export template hardened - explicit Kalpurush (Bangla) +
      // Scheherazade/Noto Naskh Arabic (for local dir="rtl" spans, item #64/
      // Phase 3 consistency) fonts, controlled ~850px width, per-page aspect
      // ratio, print break-after-per-section, and non-overlapping section
      // markers/separation, while preserving the existing section/marker
      // architecture (protected items #46-#51).
      const exportFontFamily = targetLang === 'bn' ? "'Kalpurush', sans-serif" : "'Times New Roman', Times, serif";
      const exportFontLink = targetLang === 'bn'
        ? '<link href="https://fonts.maateen.me/kalpurush/font.css" rel="stylesheet">'
        : '';
      let htmlContent = `
        <!DOCTYPE html>
        <html lang="${targetLang}">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Mirrored Translation - ${fileData?.name || 'Document'}</title>
          ${exportFontLink}
          <link href="https://fonts.googleapis.com/css2?family=Scheherazade+New:wght@400;600;700&display=swap" rel="stylesheet">
          <link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap" rel="stylesheet">
          <style>
            /* PHASE 5: target-language font applies by default, but is overridden
               for any explicitly marked local RTL span (Phase 3 embedded-quotation
               convention) so it renders in its own Arabic/Urdu typeface rather
               than being forced into the target font - no global RTL/LTR string
               hack, this is a scoped CSS rule keyed off the same dir="rtl" marker
               the prompt already asks the model to emit. */
            body, .translated-section, .translated-section * {
              font-family: ${exportFontFamily};
            }
            .translated-section [dir="rtl"] {
              font-family: 'Scheherazade New', 'Noto Naskh Arabic', serif !important;
            }
            body {
              background-color: #f8fafc;
              color: #0f172a;
              line-height: 1.8;
              padding: 2.5rem 1rem;
              margin: 0;
            }
            /* PHASE 5: each PDF page becomes its own centered, width-controlled
               "page container" with its own layout context - no overlap between
               page 1 and page 2 content, source aspect ratio respected where
               known, without forcing a fixed height that would clip overflow. */
            .page-container {
              max-width: 850px;
              margin: 0 auto 2.5rem auto;
              background: #ffffff;
              border: 1px solid #e2e8f0;
              border-radius: 12px;
              box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.08);
              padding: 2.25rem 2rem;
              box-sizing: border-box;
              overflow: visible;
              break-inside: avoid;
              page-break-inside: avoid;
            }
            .page-container-inner {
              direction: ltr;
            }
            .section-marker-wrap {
              max-width: 850px;
              margin: 0 auto 2.5rem auto;
              display: flex;
              align-items: center;
              gap: 12px;
            }
            .section-marker-wrap .line {
              flex: 1;
              border-bottom: 2px dashed #cbd5e1;
            }
            .section-marker {
              color: #94a3b8;
              font-size: 12px;
              font-weight: bold;
              font-family: 'Plus Jakarta Sans', sans-serif;
              white-space: nowrap;
            }
            @media print {
              body { background: white; padding: 0; }
              .page-container {
                border: none;
                box-shadow: none;
                border-radius: 0;
                margin: 0 auto;
                break-after: page;
                page-break-after: always;
              }
              .section-marker-wrap { display: none; }
            }
          </style>
        </head>
        <body>
      `;

      parsedSections.forEach((s) => {
        if (s.translationStatus !== 'done' || !s.translatedHtml || !s.translatedHtml.trim()) return;
        // PHASE 5: per-page aspect ratio (Phase 2 geometry) applied as a CSS
        // hint via min-height so the exported container roughly mirrors the
        // source page's proportions without clipping longer content.
        const aspectRatio = s.structure?.pageGeometry?.aspectRatio;
        const minHeightStyle = aspectRatio ? `min-height: calc(min(786px, 100vw - 4rem) / ${aspectRatio});` : '';
        htmlContent += `
          <div class="page-container" id="section-${s.id}" style="${minHeightStyle}">
            <div class="page-container-inner translated-section">
              ${sanitizeHtml(s.translatedHtml)}
            </div>
          </div>
          <div class="section-marker-wrap">
            <span class="line"></span>
            <span class="section-marker">SECTION ${s.id}</span>
            <span class="line"></span>
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

  // PHASE 5: exportHtml is now the entry point wired to the Export button. It
  // runs non-destructive structural validation first; if issues are found it
  // opens a warnings modal so the user can see them and decide whether to fix
  // pages first or export anyway, instead of silently exporting or silently
  // dropping questionable content.
  const exportHtml = () => {
    if (isExporting) return;
    const warnings = validateExportStructure(parsedSections);
    if (warnings.length > 0) {
      setExportWarnings(warnings);
      setShowExportWarningsModal(true);
      return;
    }
    performExport();
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

  const targetLabel = TARGET_LABELS[targetLang];
  const activePage = parsedSections.find(s => s.id === activeSectionId);
  const activeDiagnostics = activePage ? computeDiagnostics(activePage) : null; // PHASE 4
  const activeAspectRatio = activePage?.structure?.pageGeometry?.aspectRatio || null; // PHASE 4
  const activeSourceImage = activeSectionId ? sourcePageImages[activeSectionId] : null; // PHASE 4

  return (
    <div className="h-screen flex flex-col bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-950">

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

        <div className="flex items-center gap-3">
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
                  Key 1 is required. Keys 2-5 are optional and rotate in automatically once the earlier key hits its usage limit. Rotation cycles 1 → 2 → 3 → 4 → 5 → back to 1, forever. Minimum {REQUEST_INTERVAL_MS / 1000}s between every request.
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
                  Each page's 100% Mirrored Layout appears as soon as it finishes. Click directly into the text to make small manual edits any time. Use <strong>Compare</strong> or <strong>Overlay</strong> mode to check alignment against the original page.
                </div>
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                  <span className="font-bold text-amber-600 text-[10px] uppercase block mb-1">3. Re-translate</span>
                  Not happy with a page? Click <strong>"Re-translate"</strong> on the left sidebar for that page to generate a fresh translation. If a page fails, use <strong>"Retry"</strong> to try just that page again.
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

      {/* PHASE 5: pre-export structural warnings modal - non-destructive, lets
          the user see detected issues (missing/duplicate/malformed/empty
          sections) and decide whether to export anyway or go fix pages first. */}
      {showExportWarningsModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-100">
            <div className="flex justify-between items-center mb-4 border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
                <AlertTriangle size={18} className="text-amber-500" /> Pre-Export Warnings
              </h3>
              <button onClick={() => setShowExportWarningsModal(false)} className="p-1 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer">
                <X size={18} />
              </button>
            </div>

            <p className="text-xs text-slate-500 mb-3">The following issues were detected before export. Nothing has been changed or deleted - you can export anyway, or close this and fix the pages listed below first.</p>

            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1 mb-4">
              {exportWarnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 text-xs bg-amber-50 border border-amber-100 text-amber-800 rounded-lg px-3 py-2">
                  <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                  <span>{w.message}</span>
                </div>
              ))}
            </div>

            <div className="flex justify-end gap-2 pt-3 border-t border-slate-100">
              <button
                onClick={() => setShowExportWarningsModal(false)}
                className="px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 text-xs font-bold rounded-xl cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowExportWarningsModal(false);
                  performExport();
                }}
                className="px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold rounded-xl cursor-pointer"
              >
                Export Anyway
              </button>
            </div>
          </div>
        </div>
      )}

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
                              {s.translationStatus === 'idle' ? 'Not Processed'
                                : s.translationStatus === 'loading' ? 'Translating...'
                                : s.translationStatus === 'verifying' ? 'Verifying Extraction...'
                                : s.translationStatus === 'retranslating' ? 'Retranslating...'
                                : s.translationStatus === 'failed-phase-a' ? 'Extraction Failed'
                                : s.translationStatus === 'failed-phase-b' ? 'Translation Failed'
                                : s.translationStatus === 'error' ? 'Failed'
                                : 'Translated'}
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5">
                          {s.translationStatus === 'done' && (
                            <span className="w-2 h-2 rounded-full bg-emerald-500" title="Translated" />
                          )}
                          {(s.translationStatus === 'loading' || s.translationStatus === 'verifying' || s.translationStatus === 'retranslating') && (
                            <Loader2 size={12} className="animate-spin text-indigo-500" />
                          )}
                          {(s.translationStatus === 'error' || s.translationStatus === 'failed-phase-a' || s.translationStatus === 'failed-phase-b') && (
                            <span className="w-2 h-2 rounded-full bg-red-500" title="Failed" />
                          )}
                          <ChevronRight size={14} className={isActive ? 'text-indigo-400' : 'text-slate-300'} />
                        </div>
                      </div>

                      {/* v7 RETRY - only appears on an actual API/parse failure (not a
                          quality issue - that's what Verify/Retranslate are for). Label
                          reflects which phase actually failed, since that's all Retry redoes. */}
                      {(s.translationStatus === 'error' || s.translationStatus === 'failed-phase-a' || s.translationStatus === 'failed-phase-b') && (
                        <div className="mt-2 pl-10 pr-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveSectionId(s.id);
                              handleRetryPage(s.id);
                            }}
                            title={s.translationStatus === 'failed-phase-b' ? "Retry Phase B only (structure is already fine, just retranslate)" : "Retry this page (both phases)"}
                            className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg bg-red-50 border border-red-200 text-red-600 hover:bg-red-100 transition-all cursor-pointer text-[10px] font-bold"
                          >
                            <RotateCcw size={11} />
                            <span>{s.translationStatus === 'failed-phase-b' ? `Retry Phase B - Page ${s.id}` : `Retry Page ${s.id}`}</span>
                          </button>
                        </div>
                      )}

                      {s.translationStatus === 'done' && (
                        <div className="mt-2.5 pl-10 pr-1 flex items-center gap-1.5">
                          {/* v7 VERIFY - Phase A (extraction) only */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveSectionId(s.id);
                              handleVerifyExtraction(s.id);
                            }}
                            disabled={s.translationStatus !== 'done' || !s.extraction}
                            title={`Verify Extraction - Phase A only (Ministral 3 14B, ~2 API requests: audit + retranslate, ${REQUEST_INTERVAL_MS / 1000}s gap)`}
                            className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-indigo-600 hover:border-indigo-300 hover:bg-indigo-50 transition-all cursor-pointer disabled:opacity-50 shrink-0"
                          >
                            <ShieldCheck size={12} />
                          </button>

                          {/* v7 RETRANSLATE - Phase B only, reuses stored extraction */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveSectionId(s.id);
                              handleRetranslate(s.id);
                            }}
                            disabled={s.translationStatus !== 'done' || !s.extraction}
                            title={`Retranslate - Phase B only, reuses existing extraction (no image, cheaper/faster, ${REQUEST_INTERVAL_MS / 1000}s gap)`}
                            className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-amber-600 hover:border-amber-300 hover:bg-amber-50 transition-all cursor-pointer disabled:opacity-50 shrink-0"
                          >
                            <Languages size={12} />
                          </button>

                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleResetTranslation(s.id);
                            }}
                            disabled={s.translationStatus !== 'done' || !s.originalTranslatedHtml}
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

            <div className="flex-1 flex flex-col overflow-hidden bg-slate-50">

              <div className="bg-white border-b border-slate-200/80 px-6 py-2.5 flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-500">Workspace:</span>
                  <span className="text-xs font-extrabold text-indigo-700 bg-indigo-50 border border-indigo-100 px-2.5 py-1 rounded-md">
                    Section {activeSectionId} Workspace
                  </span>
                </div>

                {/* PHASE 4: Target / Compare / Overlay mode switcher - additive, defaults to Target */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200">
                    <button
                      onClick={() => setReviewMode('target')}
                      className={`px-2.5 py-1 text-[10px] font-bold rounded-lg transition-all cursor-pointer flex items-center gap-1 ${reviewMode === 'target' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
                    >
                      <Eye size={11} /> Target
                    </button>
                    <button
                      onClick={() => setReviewMode('compare')}
                      className={`px-2.5 py-1 text-[10px] font-bold rounded-lg transition-all cursor-pointer flex items-center gap-1 ${reviewMode === 'compare' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
                    >
                      <Columns size={11} /> Compare
                    </button>
                    <button
                      onClick={() => setReviewMode('overlay')}
                      className={`px-2.5 py-1 text-[10px] font-bold rounded-lg transition-all cursor-pointer flex items-center gap-1 ${reviewMode === 'overlay' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
                    >
                      <Layers size={11} /> Overlay
                    </button>
                  </div>

                  {reviewMode === 'overlay' && (
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] font-bold text-slate-400 uppercase">Opacity</span>
                      <input
                        type="range"
                        min="0.1"
                        max="0.95"
                        step="0.05"
                        value={overlayOpacity}
                        onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                        className="w-24 cursor-pointer"
                      />
                    </div>
                  )}

                  {/* PHASE 4: diagnostics toggle */}
                  <button
                    onClick={() => setDiagnosticsOpenFor(prev => ({ ...prev, [activeSectionId]: !prev[activeSectionId] }))}
                    className={`px-2.5 py-1.5 text-[10px] font-bold rounded-lg border transition-all cursor-pointer flex items-center gap-1 ${diagnosticsOpenFor[activeSectionId] ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'}`}
                    title="Page quality diagnostics"
                  >
                    <Info size={11} /> Diagnostics
                    {activeDiagnostics && activeDiagnostics.warnings.length > 0 && (
                      <span className="ml-1 px-1.5 py-0.5 rounded-full bg-amber-100 text-amber-700 text-[9px]">{activeDiagnostics.warnings.length}</span>
                    )}
                  </button>
                </div>
              </div>

              {/* PHASE 4: non-blocking diagnostics panel */}
              {diagnosticsOpenFor[activeSectionId] && activeDiagnostics && (
                <div className="bg-white border-b border-slate-200/80 px-6 py-3 text-xs">
                  <div className="flex flex-wrap gap-4 mb-2">
                    <span className="text-slate-500"><strong className="text-slate-800">{activeDiagnostics.sourceBlockCount}</strong> source blocks</span>
                    <span className="text-slate-500"><strong className="text-slate-800">{activeDiagnostics.translatedBlockCount}</strong> translated blocks</span>
                    <span className="text-slate-500"><strong className="text-slate-800">{activeDiagnostics.visualAssetCount}</strong> visual assets detected</span>
                    <span className="text-slate-500"><strong className="text-slate-800">{activeDiagnostics.placeholderCount}</strong> placeholders in output</span>
                    <span className="text-slate-500">Verification: <strong className="text-slate-800">{activeDiagnostics.verificationStatus}</strong></span>
                  </div>
                  {activeDiagnostics.warnings.length === 0 ? (
                    <div className="flex items-center gap-1.5 text-emerald-700">
                      <CheckCircle2 size={12} /> No warnings detected for this page.
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {activeDiagnostics.warnings.map((w, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-amber-700">
                          <AlertTriangle size={12} className="shrink-0" /> <span>{w.message}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="flex-1 flex flex-col overflow-hidden p-6 gap-6 phase4-workspace-bg">

                {(() => {
                  if (!activePage) return null;

                  const isTargetRenderable = activePage.translationStatus === 'done';

                  // Shared "target page" content renderer used by Target and Compare modes,
                  // and (with opacity) by Overlay mode.
                  const renderTargetPage = (opts = {}) => (
                    <div
                      className="phase4-page-shell"
                      style={{
                        aspectRatio: activeAspectRatio ? `${activeAspectRatio}` : undefined,
                        opacity: opts.opacity !== undefined ? opts.opacity : 1
                      }}
                    >
                      <div ref={opts.isPrimary ? editorContainerRef : null} className="phase4-page-inner h-full overflow-y-auto">
                        {isTargetRenderable ? (
                          <div
                            id={`translation-editor-${activeSectionId}`}
                            contentEditable={!!opts.editable}
                            suppressContentEditableWarning={true}
                            onBlur={opts.editable ? (e) => handleEditableContentBlur(activeSectionId, e) : undefined}
                            className={`mirror-flow ${targetLang === 'bn' ? 'bangla-font' : 'translated-text-font'} text-lg text-slate-800 text-left outline-none ring-offset-2 focus:ring-2 focus:ring-indigo-100 rounded-xl select-text animate-in fade-in duration-300`}
                            style={{ direction: 'ltr', border: 'none', padding: 0 }}
                            dangerouslySetInnerHTML={{ __html: sanitizeHtml(activePage.translatedHtml) }}
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center text-center text-slate-400 text-xs p-6">
                            No translated content yet for this page.
                          </div>
                        )}
                      </div>
                    </div>
                  );

                  const renderSourcePage = () => (
                    <div
                      className="phase4-page-shell"
                      style={{ aspectRatio: activeAspectRatio ? `${activeAspectRatio}` : undefined }}
                    >
                      <div className="h-full w-full flex items-center justify-center overflow-hidden rounded-lg">
                        {activeSourceImage ? (
                          <img
                            src={`data:image/jpeg;base64,${activeSourceImage}`}
                            alt={`Original page ${activeSectionId}`}
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <div className="flex flex-col items-center gap-2 text-slate-400 text-xs p-6">
                            {loadingSourceImage ? <Loader2 size={20} className="animate-spin" /> : <ImageOff size={20} />}
                            <span>{loadingSourceImage ? 'Rendering original page…' : 'Original page image unavailable'}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );

                  return (
                    <div className="flex-1 flex flex-col bg-white/40 rounded-xl overflow-hidden">
                      <div className="px-5 py-3 border-b border-slate-200/60 flex items-center justify-between bg-white/70">
                        <span className="text-xs font-extrabold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                          <Globe size={14} className="text-indigo-500" />
                          {reviewMode === 'target' && `100% Mirrored Target Layout (${targetLabel})`}
                          {reviewMode === 'compare' && `Compare: Original vs ${targetLabel} Translation`}
                          {reviewMode === 'overlay' && `Overlay: Original underneath, ${targetLabel} above`}
                        </span>
                        <div className="flex items-center gap-2">
                          {activePage?.translationStatus === 'done' && reviewMode === 'target' && (
                            <>
                              <span className="text-[9px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-100 px-2 py-0.5 rounded-md flex items-center gap-1">
                                <CheckCircle2 size={10} /> Translated
                              </span>
                              <button
                                onClick={() => handleCopyText(activePage.translatedHtml, `bangla-${activeSectionId}`)}
                                className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-600 transition-all cursor-pointer"
                                title={`Copy ${targetLabel} Text`}
                              >
                                {copiedId === `bangla-${activeSectionId}` ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
                              </button>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="flex-1 overflow-y-auto p-6 md:p-10">
                        {activePage.translationStatus === 'idle' && reviewMode === 'target' && (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-10 h-10 bg-slate-50 text-slate-400 rounded-full flex items-center justify-center">
                              <Globe size={18} />
                            </div>
                            <div className="max-w-xs space-y-1">
                              <h4 className="text-xs font-bold text-slate-800">Not Yet Translated</h4>
                              <p className="text-[11px] text-slate-400">Click <strong>"Translate Entire Document"</strong> to generate the {targetLabel} mirrored translation for this page.</p>
                            </div>
                          </div>
                        )}

                        {(activePage.translationStatus === 'loading' || activePage.translationStatus === 'retranslating') && reviewMode === 'target' && (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-8 h-8 border-3 border-indigo-100 rounded-full animate-spin border-t-indigo-600"></div>
                            <div className="space-y-1">
                              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">{activePage.translationStatus === 'retranslating' ? `Retranslating to ${targetLabel}...` : `Translating to ${targetLabel}...`}</p>
                              <p className="text-[10px] text-slate-400">{activePage.translationStatus === 'retranslating' ? 'Phase B only - reusing the existing page structure' : 'Phase A: reading structure, then Phase B: translating'}</p>
                            </div>
                          </div>
                        )}

                        {activePage.translationStatus === 'verifying' && reviewMode === 'target' && (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
                            <div className="w-8 h-8 border-3 border-indigo-100 rounded-full animate-spin border-t-indigo-600"></div>
                            <div className="space-y-1">
                              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Verifying Extraction...</p>
                              <p className="text-[10px] text-slate-400">Auditing the page structure (Phase A) against the original source</p>
                            </div>
                          </div>
                        )}

                        {(activePage.translationStatus === 'error' || activePage.translationStatus === 'failed-phase-a' || activePage.translationStatus === 'failed-phase-b') && reviewMode === 'target' && (
                          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-2">
                            <AlertCircle size={28} className="text-red-500" />
                            <div className="max-w-xs space-y-1">
                              <h4 className="text-xs font-bold text-slate-800">
                                {activePage.translationStatus === 'failed-phase-a' ? 'Extraction Failed (Phase A)' : activePage.translationStatus === 'failed-phase-b' ? 'Translation Failed (Phase B)' : 'Translation Failed'}
                              </h4>
                              <p className="text-[11px] text-slate-400">
                                {activePage.translationError || `Error translating content to ${targetLabel}. Please retry.`}
                              </p>
                            </div>
                            <button
                              onClick={() => handleRetryPage(activePage.id)}
                              className="mt-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-xl text-xs font-bold flex items-center gap-2 transition-all shadow-sm cursor-pointer"
                            >
                              <RotateCcw size={13} />
                              <span>{activePage.translationStatus === 'failed-phase-b' ? 'Retry Phase B Only' : 'Retry This Page'}</span>
                            </button>
                          </div>
                        )}

                        {reviewMode === 'target' && (activePage.translationStatus === 'done') && (
                          <div>
                            <p className="text-[10px] text-slate-400 mb-3 flex items-center gap-1">
                              💡 <strong>Interactive Editor:</strong> Click inside to make manual edits directly - changes save automatically.
                            </p>
                            {renderTargetPage({ editable: true, isPrimary: true })}
                          </div>
                        )}

                        {/* PHASE 4: Compare mode - original page + translation side by side */}
                        {reviewMode === 'compare' && (
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                            <div>
                              <p className="text-[10px] font-bold text-slate-400 uppercase mb-2 text-center">Original Page</p>
                              {renderSourcePage()}
                            </div>
                            <div>
                              <p className="text-[10px] font-bold text-slate-400 uppercase mb-2 text-center">{targetLabel} Translation</p>
                              {renderTargetPage({ editable: true, isPrimary: true })}
                            </div>
                          </div>
                        )}

                        {/* PHASE 4: Overlay mode - source underneath, translation above with adjustable opacity */}
                        {reviewMode === 'overlay' && (
                          <div className="relative mx-auto" style={{ maxWidth: 880 }}>
                            <div className="relative">
                              <div style={{ position: 'relative' }}>
                                {renderSourcePage()}
                                <div style={{ position: 'absolute', inset: 0 }}>
                                  {renderTargetPage({ editable: false, isPrimary: true, opacity: overlayOpacity })}
                                </div>
                              </div>
                            </div>
                            <p className="text-[10px] text-slate-400 text-center mt-3">Adjust the opacity slider above to compare block positions, missing content, and spacing between the original and the {targetLabel} translation.</p>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

              </div>
            </div>

          </div>
        )}
      </main>

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

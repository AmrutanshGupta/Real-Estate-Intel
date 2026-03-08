import os
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.config import Config
from app.logger import logger

# Pre-compile regex patterns for faster execution during the cleaning loop
_RE_ARTIFACTS = re.compile(r'[#\\|]{2,}')
_RE_NULLS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_SPACES = re.compile(r'\s+')

def _clean_text(text):
    text = _RE_ARTIFACTS.sub(' ', text)
    text = _RE_NULLS.sub('', text)
    text = _RE_SPACES.sub(' ', text)
    return text.strip()

def _readability_score(text):
    if not text:
        return 0
    printable = sum(1 for c in text if c.isprintable() and ord(c) < 256)
    alpha = sum(1 for c in text if c.isalpha())
    total = max(len(text), 1)
    return (printable / total) * 0.4 + (alpha / total) * 0.6

def _extract_page_text(page):
    # FAST PASS: Standard extraction
    text_fast = page.get_text("text").strip()
    score_fast = _readability_score(text_fast)
    
    # Short-circuit: If standard extraction is highly readable, skip the heavy processing
    if score_fast > 0.75:
        return text_fast
        
    # FALLBACK: Layout-aware blocks extraction for complex brochure formatting
    blocks = page.get_text("blocks")
    text_fallback = " ".join(b[4].strip() for b in blocks if b[4].strip() and len(b[4].strip()) > 2)
    score_fallback = _readability_score(text_fallback)
    
    # Return the best viable option, or None if the page is entirely garbled
    best_text = text_fallback if score_fallback > score_fast else text_fast
    best_score = max(score_fast, score_fallback)
    
    if best_score < 0.3:
        return None
        
    return best_text

def load_pdf(file_path):
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > Config.MAX_PDF_MB:
        logger.warning(f"Skipping {path.name}: {size_mb:.1f} MB limits exceeded.")
        return None

    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.error(f"Cannot open {path.name}: {exc}")
        return None

    pages, skipped, garbled = [], 0, 0
    total_pages = min(len(doc), Config.MAX_PAGES)

    for i in range(total_pages):
        page = doc[i]
        text = _extract_page_text(page)

        if text is None:
            garbled += 1
            continue

        text = _clean_text(text)

        if len(text) < Config.MIN_TEXT_CHARS:
            skipped += 1
            continue

        pages.append({
            "text": text,
            "page_num": i + 1,
            "source": path.name,
            "file_path": str(path),
        })

    logger.info(
        f"Loaded '{path.name}': {len(pages)} pages extracted, "
        f"{skipped} skipped (blank), {garbled} skipped (garbled font)"
    )
    return pages if pages else None
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.config import Config
from app.logger import logger


def _clean_text(text):
    """Remove font artifacts and normalize whitespace."""
    # Remove repeated special chars from embedded font artifacts (##, \\, |||)
    text = re.sub(r'[#\\|]{2,}', ' ', text)
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize multiple spaces and newlines into one space
    text = re.sub(r'\s+', ' ', text)
    # Remove lone single non-vowel characters (usually noise)
    text = re.sub(r'\s[^aeiouAEIOU\s]\s', ' ', text)
    return text.strip()


def _extract_page_text(page):
    """
    Try multiple extraction strategies and return the best result.

    Real estate brochures often use embedded/custom fonts that scramble
    raw text extraction. We try 4 methods and pick the most readable one.
    """
    results = []

    # Method 1: Standard text extraction
    text1 = page.get_text("text").strip()
    results.append(text1)

    # Method 2: HTML extraction — better at handling font mappings
    html  = page.get_text("html")
    text2 = re.sub(r'<[^>]+>', ' ', html)
    text2 = re.sub(r'&nbsp;', ' ', text2)
    text2 = re.sub(r'&#\d+;', '', text2)
    text2 = re.sub(r'\s+', ' ', text2).strip()
    results.append(text2)

    # Method 3: Blocks extraction — reads text block by block in layout order
    blocks = page.get_text("blocks")
    text3  = " ".join(
        b[4].strip() for b in blocks
        if b[4].strip() and len(b[4].strip()) > 2
    )
    results.append(text3)

    # Method 4: Dict extraction — most granular, captures spans individually
    data  = page.get_text("dict")
    spans = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    spans.append(t)
    text4 = " ".join(spans)
    results.append(text4)

    # Score each result: prefer readable ASCII/Latin text over garbled symbols
    def readability_score(text):
        if not text:
            return 0
        printable = sum(1 for c in text if c.isprintable() and ord(c) < 256)
        alpha     = sum(1 for c in text if c.isalpha())
        total     = max(len(text), 1)
        return (printable / total) * 0.4 + (alpha / total) * 0.6

    best = max(results, key=readability_score)

    # If even the best result is mostly garbage, skip the page entirely
    if readability_score(best) < 0.3:
        return None

    return best


def load_pdf(file_path):
    """
    Extract text from a PDF, returning one dict per usable page.
    Uses multi-strategy extraction to handle brochures with custom fonts.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > Config.MAX_PDF_MB:
        logger.warning(f"Skipping {path.name}: {size_mb:.1f} MB > {Config.MAX_PDF_MB} MB limit")
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

        # Clean artifacts before length check
        text = _clean_text(text)

        if len(text) < Config.MIN_TEXT_CHARS:
            skipped += 1
            continue

        pages.append({
            "text":      text,
            "page_num":  i + 1,
            "source":    path.name,
            "file_path": str(path),
        })

    logger.info(
        f"Loaded '{path.name}': {len(pages)} pages extracted, "
        f"{skipped} skipped (blank), {garbled} skipped (garbled font)"
    )
    return pages if pages else None
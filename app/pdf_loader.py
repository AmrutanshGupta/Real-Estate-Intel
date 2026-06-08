import os
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.config import Config
from app.logger import logger


_RE_ARTIFACTS = re.compile(r'[#\\|]{2,}')
_RE_NULLS     = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_SPACES    = re.compile(r'\s+')


def _clean_text(text: str) -> str:
    text = _RE_ARTIFACTS.sub(' ', text)
    text = _RE_NULLS.sub('', text)
    text = _RE_SPACES.sub(' ', text)
    return text.strip()


def _readability_score(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() and ord(c) < 256)
    alpha     = sum(1 for c in text if c.isalpha())
    total     = max(len(text), 1)
    return (printable / total) * 0.4 + (alpha / total) * 0.6


def _extract_page_text(page) -> str | None:
    text_fast  = page.get_text("text").strip()
    score_fast = _readability_score(text_fast)

    if score_fast > 0.75:
        return text_fast

    blocks        = page.get_text("blocks")
    text_fallback = " ".join(
        b[4].strip() for b in blocks if b[4].strip() and len(b[4].strip()) > 2
    )
    score_fallback = _readability_score(text_fallback)

    best_text  = text_fallback if score_fallback > score_fast else text_fast
    best_score = max(score_fast, score_fallback)

    return best_text if best_score >= 0.3 else None


def load_pdf(
    file_path: str,
    org_id: str = "default",
    doc_id: str | None = None,
) -> list[dict] | None:
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > Config.MAX_PDF_MB:
        logger.warning(f"Skipping {path.name}: {size_mb:.1f} MB exceeds limit.")
        return None

    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.error(f"Cannot open {path.name}: {exc}")
        return None

    pages, skipped, garbled = [], 0, 0
    total_pages = min(len(doc), Config.MAX_PAGES)
    _doc_id     = doc_id or path.stem

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
            "text":      text,
            "page_num":  i + 1,
            "source":    path.name,
            "file_path": str(path),
            "org_id":    org_id,
            "doc_id":    _doc_id,
        })

    logger.info(
        f"Loaded PDF '{path.name}': {len(pages)} pages extracted, "
        f"{skipped} skipped (blank), {garbled} skipped (garbled)",
        extra={"org_id": org_id, "doc_id": _doc_id},
    )
    return pages if pages else None


def load_docx(
    file_path: str,
    org_id: str = "default",
    doc_id: str | None = None,
) -> list[dict] | None:
    try:
        from docx import Document
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return None

    path    = Path(file_path)
    _doc_id = doc_id or path.stem

    try:
        doc = Document(path)
    except Exception as exc:
        logger.error(f"Cannot open DOCX {path.name}: {exc}")
        return None

    pages, para_buffer, page_num = [], [], 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        para_buffer.append(text)

        if len(para_buffer) >= 40:
            combined = _clean_text(" ".join(para_buffer))
            if len(combined) >= Config.MIN_TEXT_CHARS:
                pages.append({
                    "text":      combined,
                    "page_num":  page_num,
                    "source":    path.name,
                    "file_path": str(path),
                    "org_id":    org_id,
                    "doc_id":    _doc_id,
                })
            para_buffer = []
            page_num   += 1

    if para_buffer:
        combined = _clean_text(" ".join(para_buffer))
        if len(combined) >= Config.MIN_TEXT_CHARS:
            pages.append({
                "text":      combined,
                "page_num":  page_num,
                "source":    path.name,
                "file_path": str(path),
                "org_id":    org_id,
                "doc_id":    _doc_id,
            })

    logger.info(
        f"Loaded DOCX '{path.name}': {len(pages)} page-blocks",
        extra={"org_id": org_id, "doc_id": _doc_id},
    )
    return pages if pages else None


def load_document(
    file_path: str,
    org_id: str = "default",
    doc_id: str | None = None,
) -> list[dict] | None:
    """
    Auto-detects file type and dispatches to the correct loader.
    Returns list of page dicts, or None if loading fails.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path, org_id, doc_id)
    # 4. Strict handling to prevent crashing on legacy .doc formats
    elif ext == ".docx":
        return load_docx(file_path, org_id, doc_id)
    elif ext == ".doc":
        logger.error(f"Legacy .doc format is not supported by python-docx. Please convert '{file_path}' to .docx")
        return None
    else:
        logger.error(f"Unsupported file type: '{ext}' — only PDF and DOCX are accepted.")
        return None
#!/usr/bin/env python3
"""
ingest.py — CLI batch ingestion script (dev / data-prep tool).

Loads all PDFs from a directory and indexes them under a given org_id.
In production, ingestion happens via POST /api/upload. Use this script
to bulk-load existing data during development or migration.

Usage:
    python ingest.py                          # uses DATA_DIR, org_id=default
    python ingest.py --data ./my_pdfs         # custom directory
    python ingest.py --org acme_realty        # custom org
    python ingest.py --data ./pdfs --org acme # both
"""

import argparse
import glob
import os
import sys
import time

import torch
import faiss
from sentence_transformers import SentenceTransformer

from app.config import Config
from app.chunker import chunk_text
from app.pdf_loader import load_document
from app.vector_db import get_db, invalidate_db
from app.logger import log_ingest


def _banner(msg: str) -> None:
    print(f"\n  {msg}")


def main(data_dir: str = Config.DATA_DIR, org_id: str = "default") -> None:
    print("\n" + "━" * 60)
    print(f"  {Config.PROJECT_NAME} v{Config.VERSION} — Batch Ingestion")
    print(f"  Org: {org_id}")
    print("━" * 60)

    # ── Find files ─────────────────────────────────────────────────────────────
    pdf_files  = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    docx_files = sorted(glob.glob(os.path.join(data_dir, "*.docx")))
    files      = pdf_files + docx_files

    _banner(f"[1/4] Found {len(files)} file(s) in {data_dir}")

    if not files:
        print(f"\n  [!] No PDF or DOCX files found in {data_dir}")
        sys.exit(1)

    for f in files:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"       - {os.path.basename(f)} ({mb:.1f} MB)")

    # ── Extract text ───────────────────────────────────────────────────────────
    _banner("[2/4] Extracting text ...")

    all_pages = []
    for path in files:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        pages  = load_document(os.path.normpath(path), org_id=org_id, doc_id=doc_id)
        if pages:
            all_pages.extend(pages)
            print(f"       [OK]   {os.path.basename(path)}: {len(pages)} pages")
        else:
            print(f"       [SKIP] {os.path.basename(path)}: no extractable text")

    if not all_pages:
        print("\n  [!] No text extracted from any file. Scanned PDFs need OCR.\n")
        sys.exit(1)

    print(f"\n       Total pages: {len(all_pages)}")

    # ── Chunk ──────────────────────────────────────────────────────────────────
    _banner(
        f"[3/4] Chunking "
        f"(size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}) ..."
    )
    chunks = chunk_text(all_pages)
    print(f"       Created {len(chunks)} chunks from {len(files)} file(s)")

    # ── Embed + index ──────────────────────────────────────────────────────────
    _banner(f"[4/4] Embedding with {Config.MODEL_NAME} ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"       Device: {device.upper()}")

    model = SentenceTransformer(Config.MODEL_NAME, device=device)
    t0    = time.perf_counter()

    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=Config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)

    embed_ms = (time.perf_counter() - t0) * 1000

    if device == "cuda":
        torch.cuda.empty_cache()

    # Write to the tenant's index
    db = get_db(org_id)
    db.build(embeddings, chunks)
    invalidate_db(org_id)   # clear registry so API picks up the new index

    log_ingest(
        org_id     = org_id,
        doc_id     = f"batch_{len(files)}_files",
        chunks     = len(chunks),
        latency_ms = embed_ms,
    )

    print(f"\n  [DONE] Embedded in {embed_ms / 1000:.1f}s")
    print(f"         Vectors in index : {db.index.ntotal if db.index else 0}")
    print(f"         Saved to         : indexes/{org_id}/")
    print("━" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest PDFs/DOCX into tenant index")
    parser.add_argument(
        "--data",
        default=Config.DATA_DIR,
        help="Directory containing PDF/DOCX files",
    )
    parser.add_argument(
        "--org",
        default="default",
        help="Tenant org_id to index under (default: 'default')",
    )
    args = parser.parse_args()
    main(args.data, args.org)
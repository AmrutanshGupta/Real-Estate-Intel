#!/usr/bin/env python3
"""
ingest.py — Batch-process all PDFs in /data and build the FAISS index.

Run this once after adding PDFs.  Re-run any time the corpus changes.
The index is always rebuilt from scratch (no incremental update needed at
this scale — it takes <30 s for a typical corpus).

Usage:
    python ingest.py
    python ingest.py --data ./my_pdfs
"""

import argparse
import glob
import os
import sys
import time

import faiss

from app.config import Config
from app.chunker import chunk_text
from app.pdf_loader import load_pdf
from app.vector_db import VectorDB


def _banner(msg: str) -> None:
    print(f"\n  {msg}")


def main(data_dir: str = Config.DATA_DIR) -> None:
    print("\n" + "━" * 56)
    print(f"  {Config.PROJECT_NAME} — Batch Ingestion")
    print("━" * 56)

    # ── 1. Discover PDFs ─────────────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    _banner(f"[1/4] Found {len(files)} PDF(s) in {data_dir}")

    if not files:
        print(f"\n  ✗  No PDFs found in {data_dir}")
        print("     Drop some .pdf files there and retry.\n")
        sys.exit(1)

    for f in files:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"       • {os.path.basename(f)}  ({mb:.1f} MB)")

    # ── 2. Extract pages ─────────────────────────────────────────────────────
    _banner(f"[2/4] Extracting text …")
    all_pages = []
    for path in files:
        pages = load_pdf(os.path.normpath(path))
        if pages:
            all_pages.extend(pages)
            print(f"       ✓  {os.path.basename(path)}: {len(pages)} pages")
        else:
            print(f"       ✗  {os.path.basename(path)}: no text extracted (scanned?)")

    if not all_pages:
        print("\n  ✗  No text from any PDF.  Scanned documents need OCR first.\n")
        sys.exit(1)

    print(f"\n       Total pages: {len(all_pages)}")

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    _banner(f"[3/4] Chunking (size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}) …")
    chunks = chunk_text(all_pages)
    print(f"       Created {len(chunks)} chunks across {len(files)} document(s)")

    # ── 4. Embed + index ─────────────────────────────────────────────────────
    _banner(f"[4/4] Embedding with {Config.MODEL_NAME} …")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(Config.MODEL_NAME)

    t0         = time.perf_counter()
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=Config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embed_ms = (time.perf_counter() - t0) * 1000

    faiss.normalize_L2(embeddings)

    db = VectorDB()
    db.build(embeddings, chunks)

    print(f"\n  ✓  Done in {embed_ms / 1000:.1f} s")
    print(f"     Index:  {db.index.ntotal} vectors")
    print(f"     Saved:  {Config.INDEX_DIR}")
    print("━" * 56 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Config.DATA_DIR, help="Directory of PDFs")
    args = parser.parse_args()
    main(args.data)

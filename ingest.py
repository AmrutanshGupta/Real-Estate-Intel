#!/usr/bin/env python3
"""
ingest.py — CLI batch ingestion script (dev / data-prep tool).

Loads all PDFs from a directory and indexes them under a given org_id.
Now utilizes the unified asynchronous SearchEngine pipeline.
"""

import argparse
import glob
import os
import sys
import time
import asyncio

from app.config import Config
from app.pdf_loader import load_document
from app.search_engine import get_engine, invalidate_engine
from app.vector_db import invalidate_db
from app.logger import log_ingest


def _banner(msg: str) -> None:
    print(f"\n  {msg}")


async def run_ingestion(data_dir: str, org_id: str) -> None:
    print("\n" + "━" * 60)
    print(f"  {Config.PROJECT_NAME} v{Config.VERSION} — Batch Ingestion")
    print(f"  Org: {org_id}")
    print("━" * 60)

    # ── Find files ─────────────────────────────────────────────────────────────
    pdf_files  = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    docx_files = sorted(glob.glob(os.path.join(data_dir, "*.docx")))
    files      = pdf_files + docx_files

    _banner(f"[1/3] Found {len(files)} file(s) in {data_dir}")

    if not files:
        print(f"\n  [!] No PDF or DOCX files found in {data_dir}")
        sys.exit(1)

    for f in files:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"       - {os.path.basename(f)} ({mb:.1f} MB)")

    # ── Extract text ───────────────────────────────────────────────────────────
    _banner("[2/3] Extracting text (CPU Bound) ...")

    all_pages = []
    for path in files:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        # Offload to thread to mimic API behavior
        pages  = await asyncio.to_thread(load_document, os.path.normpath(path), org_id=org_id, doc_id=doc_id)
        
        if pages:
            all_pages.extend(pages)
            print(f"       [OK]   {os.path.basename(path)}: {len(pages)} pages")
        else:
            print(f"       [SKIP] {os.path.basename(path)}: no extractable text")

    if not all_pages:
        print("\n  [!] No text extracted from any file. Scanned PDFs need OCR.\n")
        sys.exit(1)

    print(f"\n       Total pages extracted: {len(all_pages)}")

    # ── Unified Pipeline (Semantic Chunking + Embed + Index) ────────────────────
    _banner(f"[3/3] Running Async Pipeline (Chunking & Embedding) ...")
    
    engine = get_engine(org_id)
    
    t0 = time.perf_counter()
    # Leverage the exact same async ingest method the API uses
    stats = await engine.ingest(all_pages)
    embed_ms = (time.perf_counter() - t0) * 1000

    invalidate_db(org_id)   # clear registry so API picks up the new index

    log_ingest(
        org_id     = org_id,
        doc_id     = f"batch_{len(files)}_files",
        chunks     = stats.get("chunks", 0),
        latency_ms = embed_ms,
    )

    print(f"\n  [DONE] Processing completed in {embed_ms / 1000:.1f}s")
    print(f"         Vectors in index : {engine.stats.get('vectors', 0)}")
    print(f"         Saved to         : indexes/{org_id}/")
    print("━" * 60 + "\n")


def main():
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
    
    # Run the async ingestion loop
    asyncio.run(run_ingestion(args.data, args.org))


if __name__ == "__main__":
    main()
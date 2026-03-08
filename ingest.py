#!/usr/bin/env python3
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
from app.pdf_loader import load_pdf
from app.vector_db import VectorDB

def _banner(msg: str) -> None:
    print(f"\n  {msg}")

def main(data_dir: str = Config.DATA_DIR) -> None:
    print("\n" + "━" * 56)
    print(f"  {Config.PROJECT_NAME} - Batch Ingestion")
    print("━" * 56)

    files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    _banner(f"[1/4] Found {len(files)} PDF(s) in {data_dir}")

    if not files:
        print(f"\n  [!] No PDFs found in {data_dir}")
        print("      Drop some .pdf files there and retry.\n")
        sys.exit(1)

    for f in files:
        mb = os.path.getsize(f) / (1024 * 1024)
        print(f"       - {os.path.basename(f)}  ({mb:.1f} MB)")

    _banner("[2/4] Extracting text ...")
    all_pages = []
    for path in files:
        pages = load_pdf(os.path.normpath(path))
        if pages:
            all_pages.extend(pages)
            print(f"       [OK] {os.path.basename(path)}: {len(pages)} pages")
        else:
            print(f"       [SKIP] {os.path.basename(path)}: no extractable text")

    if not all_pages:
        print("\n  [!] No text from any PDF. Scanned documents need OCR first.\n")
        sys.exit(1)

    print(f"\n       Total pages: {len(all_pages)}")

    _banner(f"[3/4] Chunking (size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}) ...")
    chunks = chunk_text(all_pages)
    print(f"       Created {len(chunks)} chunks across {len(files)} document(s)")

    _banner(f"[4/4] Embedding with {Config.MODEL_NAME} ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(Config.MODEL_NAME, device=device)

    t0 = time.perf_counter()
    texts = [c["text"] for c in chunks]
    
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

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  [DONE] Completed in {embed_ms / 1000:.1f} s")
    print(f"         Index:  {db.index.ntotal if db.index else 0} vectors")
    print(f"         Saved:  {Config.INDEX_DIR}")
    print("━" * 56 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Config.DATA_DIR, help="Directory of PDFs")
    args = parser.parse_args()
    main(args.data)
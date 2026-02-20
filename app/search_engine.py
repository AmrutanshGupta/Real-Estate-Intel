import math
import re
import time
import torch
from concurrent.futures import ThreadPoolExecutor

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import Config
from app.logger import logger
from app.vector_db import VectorDB

class SearchEngine:
    MIN_SCORE = 0.0

    def __init__(self):
        self.db = VectorDB()
        self.ready = False
        
        # Hardware detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto {device.upper()}...")

        self.model = SentenceTransformer(Config.MODEL_NAME, device=device)
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=512)
        except Exception as e:
            logger.warning(f"Reranker MIA: {e}. Falling back to standard scores.")
            self.reranker = None

        self._pool = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
        if self.db.load(): self.ready = True

    _SYNONYMS = {
        "price": "price cost rate value amount per sqft", "area": "area size carpet super built sqft square feet saleable",
        "parking": "parking car park garage stilt covered two wheeler", "amenities": "amenities facilities features club gym pool recreation",
        "possession": "possession handover ready completion delivery date", "rera": "rera registration number approved authority",
    }

    def _normalize_expand(self, text):
        text = text.lower()
        extra = [syn for key, syn in self._SYNONYMS.items() if key in text]
        return text + " " + " ".join(extra) if extra else text

    def query(self, text, k=Config.DEFAULT_K):
        t0 = time.perf_counter()
        if not self.ready or not text.strip(): 
            return {"results": [], "error": "Index missing or empty query.", "latency_ms": 0}

        k = min(max(1, k), Config.MAX_K)
        expanded = self._normalize_expand(text)

        # Non-blocking embedding generation
        vector = self._pool.submit(lambda: self.model.encode([expanded], convert_to_numpy=True)).result()
        faiss.normalize_L2(vector)

        candidates = self.db.hybrid_search(vector, expanded, k=k * 4)
        if not candidates: return {"results": [], "message": "No matches found."}

        if self.reranker and len(candidates) > 1:
            pairs = [(text, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            results = [{**c, "score": round(1 / (1 + math.exp(-float(s) / 3)), 4)} for s, c in ranked[:k]]
        else:
            results = candidates[:k]

        # Filter and Deduplicate
        results = [r for r in results if r["score"] >= self.MIN_SCORE]
        seen, unique = set(), []
        for r in results:
            key = (r["source"], r["page_num"], r.get("chunk_index", 0))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return {
            "results": unique, "query": text, "k": k, 
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
        }

    def ingest(self, pages):
        from app.chunker import chunk_text
        t0 = time.perf_counter()
        chunks = chunk_text(pages)
        if not chunks: return {"error": "No usable text.", "chunks": 0}

        embeddings = self.model.encode([c["text"] for c in chunks], batch_size=Config.BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        if self.db.index is None: self.db.build(embeddings, chunks)
        else: self.db.add(embeddings, chunks)

        self.ready = True
        return {"chunks": len(chunks), "ingest_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @property
    def stats(self): return {**self.db.stats, "model": Config.MODEL_NAME}
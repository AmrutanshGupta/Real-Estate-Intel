import os
import pickle
import threading
import faiss
import numpy as np

from app.config import Config
from app.logger import logger

class VectorDB:
    def __init__(self):
        self.index = None
        self.metadata = {}
        self.bm25 = None
        self._lock = threading.RLock()
        self.res = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None

    def _to_gpu(self):
        if self.index is not None and self.res is not None:
            try:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            except Exception as e:
                logger.warning(f"GPU transfer failed. Proceeding on CPU: {e}")

    def build(self, embeddings, meta_list):
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings array provided.")
        
        with self._lock:
            cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
            cpu_index.add(embeddings)
            self.index = cpu_index
            self._to_gpu()
            
            self.metadata = {i: m for i, m in enumerate(meta_list)}
            self._build_bm25()
        self._persist()

    def add(self, embeddings, meta_list):
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build() first.")
        
        with self._lock:
            base = self.index.ntotal
            self.index.add(embeddings)
            for i, m in enumerate(meta_list):
                self.metadata[base + i] = m
            self._build_bm25()
        self._persist()

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            corpus = [self.metadata[i]["text"].lower().split() for i in range(len(self.metadata))]
            self.bm25 = BM25Okapi(corpus)
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 search disabled.")
            self.bm25 = None

    def _persist(self):
        if self.index is None:
            return
            
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.res else self.index
        faiss.write_index(cpu_index, Config.INDEX_PATH)
        with open(Config.META_PATH, "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self):
        if not os.path.exists(Config.INDEX_PATH) or not os.path.exists(Config.META_PATH):
            return False
            
        with self._lock:
            self.index = faiss.read_index(Config.INDEX_PATH)
            self._to_gpu()
            with open(Config.META_PATH, "rb") as fh:
                self.metadata = pickle.load(fh)
            self._build_bm25()
        return True

    def hybrid_search(self, vector, query_text, k=5, allowed_source=None):
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # 1. THE HARD FILTER: Identify valid chunk IDs if isolating a specific property
        valid_ids = None
        if allowed_source:
            valid_ids = {i for i, m in self.metadata.items() if m.get("source") == allowed_source}
            if not valid_ids:
                return [] # Property not found in DB
                
        # 2. SEMANTIC SEARCH (FAISS)
        # If filtering, over-fetch to ensure we get enough valid candidates
        fetch_k = self.index.ntotal if allowed_source else min(k * 4, self.index.ntotal)
        scores, indices = self.index.search(vector, fetch_k)
        
        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                # Apply the Hard Filter
                if valid_ids and int(idx) not in valid_ids:
                    continue
                semantic_results.append((int(idx), float(score)))

        # 3. KEYWORD SEARCH (BM25)
        bm25_results = []
        if self.bm25:
            tokens = query_text.lower().split()
            bm25_scores = np.array(self.bm25.get_scores(tokens))
            
            # Apply the Hard Filter by zeroing out scores of other properties
            if valid_ids:
                mask = np.ones(len(bm25_scores), dtype=bool)
                mask[list(valid_ids)] = False
                bm25_scores[mask] = 0.0

            if len(bm25_scores) > k * 4:
                top_ids = np.argpartition(bm25_scores, -(k * 4))[-(k * 4):]
                top_ids = top_ids[np.argsort(bm25_scores[top_ids])[::-1]]
            else:
                top_ids = np.argsort(bm25_scores)[::-1]
                
            bm25_results = [(int(i), float(bm25_scores[i])) for i in top_ids if bm25_scores[i] > 0]

        # 4. RECIPROCAL RANK FUSION (RRF)
        rrf_scores = {}
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + 60)
            
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + 60)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [{**self.metadata[idx], "score": round(rrf_scores[idx], 4)} for idx in sorted_ids]

    @property
    def stats(self):
        if self.index is None:
            return {"vectors": 0, "ready": False}
        doc_count = len({m["source"] for m in self.metadata.values()})
        return {"vectors": self.index.ntotal, "documents": doc_count, "ready": self.index.ntotal > 0}
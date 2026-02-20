import os
import pickle
import threading
import faiss
import numpy as np

from app.config import Config
from app.logger import logger

class VectorDB:
    def __init__(self):
        self.index    = None
        self.metadata = {}
        self.bm25     = None
        self._lock    = threading.RLock()
        self.res      = faiss.StandardGpuResources() if hasattr(faiss, 'StandardGpuResources') else None

    def _to_gpu(self):
        """Moves the index to CUDA if resources permit."""
        if self.index is not None and self.res is not None:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

    def build(self, embeddings, meta_list):
        if len(embeddings) == 0: raise ValueError("0 embeddings provided.")
        with self._lock:
            # Build on CPU, then transfer
            cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
            cpu_index.add(embeddings)
            self.index = cpu_index
            self._to_gpu()
            self.metadata = {i: m for i, m in enumerate(meta_list)}
            self._build_bm25()
        self._persist()

    def add(self, embeddings, meta_list):
        if self.index is None: raise RuntimeError("Index missing.")
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
            self.bm25 = None

    def _persist(self):
        # Must drop back to CPU to save safely
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.res else self.index
        faiss.write_index(cpu_index, Config.INDEX_PATH)
        with open(Config.META_PATH, "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self):
        if not os.path.exists(Config.INDEX_PATH): return False
        with self._lock:
            self.index = faiss.read_index(Config.INDEX_PATH)
            self._to_gpu()
            with open(Config.META_PATH, "rb") as fh:
                self.metadata = pickle.load(fh)
            self._build_bm25()
        return True

    def hybrid_search(self, vector, query_text, k=5):
        if self.index is None or self.index.ntotal == 0: return []
        candidate_k = min(k * 4, self.index.ntotal)
        
        scores, indices = self.index.search(vector, candidate_k)
        semantic_results = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

        bm25_results = []
        if self.bm25:
            tokens = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(tokens)
            top_ids = np.argsort(bm25_scores)[-candidate_k:][::-1]
            bm25_results = [(int(i), float(bm25_scores[i])) for i in top_ids if bm25_scores[i] > 0]

        rrf_scores = {}
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 60)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 60)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [{**self.metadata[idx], "score": round(rrf_scores[idx], 4)} for idx in sorted_ids]

    @property
    def stats(self):
        if self.index is None: return {"vectors": 0, "ready": False}
        return {"vectors": self.index.ntotal, "documents": len({m["source"] for m in self.metadata.values()}), "ready": self.index.ntotal > 0}
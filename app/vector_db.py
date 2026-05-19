import os
import pickle
import threading

import faiss
import numpy as np

from app.config import Config
from app.logger import logger


# ── VectorDB class ─────────────────────────────────────────────────────────────

class VectorDB:
    """
    FAISS + BM25 hybrid index scoped to a single tenant (org_id).
    Each tenant gets its own directory: indexes/{org_id}/
    """

    def __init__(self, org_id: str = "default"):
        self.org_id   = org_id
        self.index    = None
        self.metadata = {}
        self.bm25     = None
        self._lock    = threading.RLock()

        # GPU resources (gracefully falls back to CPU if unavailable)
        self.res = (
            faiss.StandardGpuResources()
            if hasattr(faiss, "StandardGpuResources")
            else None
        )

        # Ensure tenant index directory exists
        os.makedirs(os.path.join(Config.INDEX_DIR, org_id), exist_ok=True)

    # ── GPU helpers ────────────────────────────────────────────────────────────

    def _to_gpu(self) -> None:
        if self.index is not None and self.res is not None:
            try:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            except Exception as e:
                logger.warning(f"GPU transfer failed, running on CPU: {e}")

    def _to_cpu(self):
        if self.res and self.index is not None:
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    # ── Index build / add ──────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, meta_list: list[dict]) -> None:
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings array provided.")

        with self._lock:
            cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
            cpu_index.add(embeddings)
            self.index    = cpu_index
            self.metadata = {i: m for i, m in enumerate(meta_list)}
            self._to_gpu()
            self._build_bm25()
            self._persist()

    def add(self, embeddings: np.ndarray, meta_list: list[dict]) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build() first.")

        with self._lock:
            base = self.index.ntotal
            self.index.add(embeddings)
            for i, m in enumerate(meta_list):
                self.metadata[base + i] = m
            self._build_bm25()
            self._persist()

    # ── BM25 ───────────────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            corpus    = [self.metadata[i]["text"].lower().split() for i in range(len(self.metadata))]
            self.bm25 = BM25Okapi(corpus)
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 search disabled.")
            self.bm25 = None

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if self.index is None:
            return
        cpu_index = self._to_cpu()
        faiss.write_index(cpu_index, Config.index_path(self.org_id))
        with open(Config.meta_path(self.org_id), "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self) -> bool:
        if (
            not os.path.exists(Config.index_path(self.org_id))
            or not os.path.exists(Config.meta_path(self.org_id))
        ):
            return False

        with self._lock:
            self.index = faiss.read_index(Config.index_path(self.org_id))
            self._to_gpu()
            with open(Config.meta_path(self.org_id), "rb") as fh:
                self.metadata = pickle.load(fh)
            self._build_bm25()

        logger.info(
            f"Index loaded for org '{self.org_id}': {self.index.ntotal} vectors",
            extra={"org_id": self.org_id},
        )
        return True

    # ── Hybrid search ──────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        vector: np.ndarray,
        query_text: str,
        k: int = 5,
        allowed_source: str | None = None,
    ) -> list[dict]:
        """
        FAISS dense + BM25 sparse search fused via Reciprocal Rank Fusion (RRF).
        allowed_source enforces the Hard Filter for per-property isolation.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # ── Hard Filter: identify valid chunk IDs for this property ───────────
        valid_ids = None
        if allowed_source:
            valid_ids = {
                i for i, m in self.metadata.items()
                if m.get("source") == allowed_source
            }
            if not valid_ids:
                return []

        # ── FAISS semantic search ──────────────────────────────────────────────
        fetch_k = self.index.ntotal if allowed_source else min(k * 4, self.index.ntotal)
        scores, indices = self.index.search(vector, fetch_k)

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if valid_ids and int(idx) not in valid_ids:
                continue
            semantic_results.append((int(idx), float(score)))

        # ── BM25 keyword search ────────────────────────────────────────────────
        bm25_results = []
        if self.bm25:
            tokens      = query_text.lower().split()
            bm25_scores = np.array(self.bm25.get_scores(tokens))

            if valid_ids:
                mask = np.zeros(len(bm25_scores), dtype=bool)
                mask[list(valid_ids)] = True
                bm25_scores[~mask] = 0.0

            candidate_k = k * 4
            if len(bm25_scores) > candidate_k:
                top_ids = np.argpartition(bm25_scores, -candidate_k)[-candidate_k:]
                top_ids = top_ids[np.argsort(bm25_scores[top_ids])[::-1]]
            else:
                top_ids = np.argsort(bm25_scores)[::-1]

            bm25_results = [
                (int(i), float(bm25_scores[i]))
                for i in top_ids
                if bm25_scores[i] > 0
            ]

        # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 60)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + 60)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        return [
            {**self.metadata[idx], "score": round(rrf_scores[idx], 4)}
            for idx in sorted_ids
        ]

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        if self.index is None:
            return {"vectors": 0, "documents": 0, "ready": False, "org_id": self.org_id}
        doc_count = len({m["source"] for m in self.metadata.values()})
        return {
            "vectors":   self.index.ntotal,
            "documents": doc_count,
            "ready":     self.index.ntotal > 0,
            "org_id":    self.org_id,
        }

# ── Per-tenant registry ────────────────────────────────────────────────────────
_registry:      dict[str, VectorDB] = {}
_registry_lock: threading.Lock      = threading.Lock()

def get_db(org_id: str) -> VectorDB:
    with _registry_lock:
        if org_id not in _registry:
            db = VectorDB(org_id)
            db.load()
            _registry[org_id] = db
        return _registry[org_id]

def invalidate_db(org_id: str) -> None:
    with _registry_lock:
        _registry.pop(org_id, None)
import math
import time
import torch
import torch.quantization
import numpy as np
import threading
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import Config
from app.logger import logger
from app.llm_layer import generate_answer


# ---------------------------------------------------------------------------
# Lightweight named-entity helpers (no spaCy dependency)
# ---------------------------------------------------------------------------

# Heuristic: a "name token" is a capitalised word of ≥2 chars that is NOT a
# common English stopword.  Good enough for proper-noun extraction from short
# real-estate queries without pulling in a 40 MB NER model.
_STOPWORDS = frozenset({
    "who", "what", "where", "when", "how", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "about", "tell", "me", "give", "show", "find", "list",
    "does", "do", "did", "has", "have", "had", "be", "been", "being",
    "this", "that", "these", "those", "my", "your", "its", "their",
    "project", "property", "flat", "apartment", "tower", "block",
})

def _extract_name_tokens(text: str) -> set[str]:
    """
    Return capitalised tokens that look like proper nouns.
    Used for entity-consistency gating — not for full NER.
    """
    tokens = re.findall(r'\b[A-Z][a-z]{1,}\b', text)
    return {t.lower() for t in tokens if t.lower() not in _STOPWORDS}


def _entity_consistency_penalty(query: str, chunk_text: str, score: float) -> float:
    """
    Post-rerank gate: if the query contains capitalised name tokens that are
    completely absent from the chunk, apply a heavy penalty.

    This is the *last* line of defence — only fires when the reranker already
    let a wrong chunk through.
    """
    name_tokens = _extract_name_tokens(query)
    if not name_tokens:
        return score          # query has no proper nouns → nothing to check

    chunk_lower = chunk_text.lower()
    if any(tok in chunk_lower for tok in name_tokens):
        return score          # at least one name token present → chunk is legit

    # Zero name tokens matched → almost certainly the wrong person/project
    return score * 0.25


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:
    def __init__(self, org_id: str = "default"):
        from app.vector_db import get_db
        self.org_id = org_id
        self.db = get_db(org_id)
        self.ready = self.db.index is not None

        self._query_cache: dict = {}
        self._embed_cache: dict = {}
        self._cache_lock = threading.Lock()
        self.MAX_CACHE_SIZE = getattr(Config, 'MAX_CACHE_SIZE', 1000)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading models onto {self.device.upper()} for org '{org_id}'...")

        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

        # INT8 quantisation on CPU — free latency win
        if self.device == "cpu":
            logger.info("Applying INT8 Dynamic Quantization to Embedding Model...")
            self.model[0].auto_model = torch.quantization.quantize_dynamic(
                self.model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self.use_reranker = getattr(Config, 'USE_RERANKER', True)
        if self.use_reranker:
            try:
                self.reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device=self.device,
                    max_length=512,
                )
                if self.device == "cpu":
                    logger.info("Applying INT8 Dynamic Quantization to Reranker...")
                    self.reranker.model = torch.quantization.quantize_dynamic(
                        self.reranker.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}. Falling back.")
                self.reranker = None
        else:
            self.reranker = None

        self._pool = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    # ------------------------------------------------------------------
    # Synonym expansion
    # ------------------------------------------------------------------

    _SYNONYMS = {
        "price":      "price cost rate value amount per sqft",
        "area":       "area size carpet super built sqft square feet saleable",
        "parking":    "parking car park garage stilt covered two wheeler",
        "amenities":  "amenities facilities features club gym pool recreation",
        "possession": "possession handover ready completion delivery date",
        "rera":       "rera registration number approved authority",
    }

    def _normalize_expand(self, text: str) -> str:
        text = text.lower()
        extra = [syn for key, syn in self._SYNONYMS.items() if key in text]
        return text + " " + " ".join(extra) if extra else text

    # ------------------------------------------------------------------
    # Entity / source helpers
    # ------------------------------------------------------------------

    def _get_active_entities(self) -> list[str]:
        if not self.db.metadata:
            return []
        return list(set(m["source"] for m in self.db.metadata.values()))

    def _detect_query_entities(self, text: str) -> list[str]:
        text_lower = text.lower()
        active_docs = self._get_active_entities()
        return [
            doc for doc in active_docs
            if doc.replace('.pdf', '').lower() in text_lower
            or doc.replace('-brochure', '').replace('_brochure', '').lower() in text_lower
        ]

    # ------------------------------------------------------------------
    # Embedding (with LRU-ish cache)
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> np.ndarray:
        with self._cache_lock:
            if text in self._embed_cache:
                return self._embed_cache[text]

        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)

        with self._cache_lock:
            if len(self._embed_cache) >= self.MAX_CACHE_SIZE:
                self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[text] = vector
        return vector

    def _encode_batch(self, sentences: list[str]) -> np.ndarray:
        """Encode a list of sentences — passed as encode_fn to the chunker."""
        return self.model.encode(sentences, convert_to_numpy=True)

    # ------------------------------------------------------------------
    # MMR diversity filter
    # ------------------------------------------------------------------

    def _apply_mmr(self, candidates, query_vec, k: int, lambda_param: float = 0.7):
        if not candidates or len(candidates) <= 1:
            return candidates[:k]

        selected = []
        cand_embeds = np.vstack([self._get_embedding(c["text"]) for c in candidates])
        scores_q = np.dot(cand_embeds, query_vec.T).flatten()
        remaining = list(range(len(candidates)))

        best = int(np.argmax(scores_q))
        selected.append(best)
        remaining.remove(best)

        while len(selected) < k and remaining:
            mmr_scores = [
                lambda_param * scores_q[idx]
                - (1 - lambda_param) * max(np.dot(cand_embeds[idx], cand_embeds[s]) for s in selected)
                for idx in remaining
            ]
            best = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Score (query, chunk) pairs with the Cross-Encoder, apply temperature
        scaling for calibration, then filter with a meaningful threshold.

        Threshold rationale
        -------------------
        ms-marco-MiniLM-L-6-v2 raw logits are roughly:
          > 0   → some relevance
          > 5   → clearly relevant
        After sigmoid with temperature=2 we want:
          keep  ≥ 0.30  (roughly logit > 1.4 raw — only genuinely relevant chunks)
          drop  < 0.30

        0.15 was far too permissive; it kept chunks that are "kinda topically
        similar" but wrong.  0.30 is conservative enough for proper nouns.
        """
        TEMPERATURE      = 2.0
        KEEP_THRESHOLD   = 0.30   # was 0.15 — raised to cut wrong-entity chunks
        MAX_RESULTS      = 5

        pairs  = [(query, c["text"]) for c in candidates]
        logits = np.array(self.reranker.predict(pairs), dtype=float)

        # Sigmoid with temperature scaling (single normalisation — no double-norm)
        probs = 1.0 / (1.0 + np.exp(-logits / TEMPERATURE))

        ranked = sorted(zip(probs, candidates), key=lambda x: x[0], reverse=True)

        filtered = []
        for prob, chunk in ranked:
            if prob < KEEP_THRESHOLD:
                break                               # sorted descending → safe to break

            # Entity-consistency gate: last-resort penalty for wrong-person chunks
            penalised = _entity_consistency_penalty(query, chunk["text"], float(prob))

            if penalised >= KEEP_THRESHOLD:
                chunk["score"] = round(penalised * 100, 1)
                filtered.append(chunk)

        return filtered[:MAX_RESULTS]

    def _fallback_rank(self, query: str, candidates: list[dict], query_vec: np.ndarray) -> list[dict]:
        """
        Used when the reranker is unavailable.  Pure RRF score (from hybrid_search)
        is already a good signal — we just normalise to 0–100 and apply the entity
        gate.  No invented "blend" that duplicates BM25 work.
        """
        KEEP_THRESHOLD = 0.35
        MAX_RESULTS    = 5

        raw_scores = np.array([c.get("score", 0.01) for c in candidates], dtype=float)
        max_s = raw_scores.max() if raw_scores.max() > 0 else 1.0
        norm_scores = raw_scores / max_s          # 0–1, relative to best candidate

        ranked = sorted(
            zip(norm_scores, candidates), key=lambda x: x[0], reverse=True
        )

        filtered = []
        for norm_s, chunk in ranked:
            penalised = _entity_consistency_penalty(query, chunk["text"], float(norm_s))
            if penalised >= KEEP_THRESHOLD:
                chunk["score"] = round(penalised * 100, 1)
                filtered.append(chunk)

        return filtered[:MAX_RESULTS]

    # ------------------------------------------------------------------
    # Main async query entry-point
    # ------------------------------------------------------------------

    async def query(self, text: str, k: int = Config.DEFAULT_K) -> dict:
        t_start = time.perf_counter()
        latencies = {"embedding": 0.0, "retrieval": 0.0, "reranking": 0.0, "llm": 0.0, "total": 0.0}

        if not self.ready or not text.strip():
            return {"results": [], "error": "Index missing.", "latency": latencies}

        k = min(max(1, k), Config.MAX_K)
        cache_key = f"{self.org_id}:{text.strip().lower()}:{k}"

        if getattr(Config, 'USE_QUERY_CACHE', True):
            with self._cache_lock:
                if cache_key in self._query_cache:
                    cached = self._query_cache[cache_key].copy()
                    latencies["total"] = round((time.perf_counter() - t_start) * 1000, 2)
                    cached["latency"] = latencies
                    cached["cached"]  = True
                    return cached

        detected_entities = self._detect_query_entities(text)
        expanded = self._normalize_expand(text)

        # ── Embedding ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vec = await asyncio.to_thread(self._get_embedding, expanded)
        latencies["embedding"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Retrieval (hybrid BM25 + FAISS) ───────────────────────────────
        t0 = time.perf_counter()
        if len(detected_entities) > 1:
            per_entity_k = max(k, 5)
            candidates: list[dict] = []
            for entity in detected_entities:
                hits = await asyncio.to_thread(
                    self.db.hybrid_search, query_vec, expanded, k=per_entity_k, allowed_source=entity
                )
                candidates.extend(hits)
        else:
            candidates = await asyncio.to_thread(
                self.db.hybrid_search, query_vec, expanded, k=k * 4
            )
        latencies["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        if not candidates:
            return {"results": [], "message": "No matches found.", "latency": latencies}

        # De-duplicate
        seen, unique_cands = set(), []
        for c in candidates:
            uid = (c.get("source"), c.get("page_num"), c.get("chunk_index", 0))
            if uid not in seen:
                seen.add(uid)
                unique_cands.append(c)

        # ── Reranking / scoring ───────────────────────────────────────────
        t0 = time.perf_counter()
        if self.reranker and len(unique_cands) > 1:
            filtered = await asyncio.to_thread(self._rerank, text, unique_cands)
        else:
            filtered = await asyncio.to_thread(self._fallback_rank, text, unique_cands, query_vec)
        latencies["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── MMR diversity + final sort ────────────────────────────────────
        final_results = self._apply_mmr(filtered, query_vec, len(filtered))
        final_results = sorted(final_results, key=lambda x: x.get("score", 0), reverse=True)

        # ── LLM generation ───────────────────────────────────────────────
        t0 = time.perf_counter()
        query_type = "comparison" if len(detected_entities) > 1 else "general"
        llm_resp = await asyncio.to_thread(
            generate_answer,
            query=text,
            chunks=final_results,
            query_type=query_type,
            entities=detected_entities,
        )
        latencies["llm"]   = round((time.perf_counter() - t0) * 1000, 2)
        latencies["total"] = round((time.perf_counter() - t_start) * 1000, 2)

        response = {
            "answer":            llm_resp.answer,
            "sources":           llm_resp.sources,
            "refused":           llm_resp.refused,
            "results":           final_results,
            "query":             text,
            "query_type":        query_type,
            "entities_detected": detected_entities,
            "latency":           latencies,
            "cached":            False,
        }

        if getattr(Config, 'USE_QUERY_CACHE', True):
            with self._cache_lock:
                if len(self._query_cache) >= self.MAX_CACHE_SIZE:
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[cache_key] = response

        return response

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest(self, pages: list[dict]) -> dict:
        from app.chunker import chunk_text
        t0 = time.perf_counter()

        with self._cache_lock:
            self._query_cache.clear()
            self._embed_cache.clear()

        # Pass encode_fn so the chunker uses *our* already-loaded model
        chunks = await asyncio.to_thread(
            chunk_text, pages, 0.45, self._encode_batch
        )
        if not chunks:
            return {"error": "No text found.", "chunks": 0}

        embeddings = await asyncio.to_thread(
            self.model.encode,
            [c["text"] for c in chunks],
            batch_size=Config.BATCH_SIZE,
            convert_to_numpy=True,
        )
        faiss.normalize_L2(embeddings)

        if self.db.index is None:
            await asyncio.to_thread(self.db.build, embeddings, chunks)
        else:
            await asyncio.to_thread(self.db.add, embeddings, chunks)

        self.ready = True
        return {"chunks": len(chunks), "ingest_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {**self.db.stats, "model": Config.MODEL_NAME, "org_id": self.org_id}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_engine_registry: dict[str, SearchEngine] = {}
_engine_lock = threading.Lock()


def get_engine(org_id: str) -> SearchEngine:
    with _engine_lock:
        if org_id not in _engine_registry:
            _engine_registry[org_id] = SearchEngine(org_id)
        return _engine_registry[org_id]


def invalidate_engine(org_id: str) -> None:
    with _engine_lock:
        _engine_registry.pop(org_id, None)
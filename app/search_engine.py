import math
import time
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import Config
from app.logger import logger
from app.llm_layer import generate_answer

class SearchEngine:
    def __init__(self, org_id: str = "default"):
        from app.vector_db import get_db
        self.org_id = org_id
        self.db = get_db(org_id)
        self.ready = self.db.index is not None
        
        # 1. Thread-safe custom caching (Replaces leaky @lru_cache)
        self._query_cache = {}
        self._embed_cache = {} 
        self._cache_lock = threading.Lock() 
        self.MAX_CACHE_SIZE = getattr(Config, 'MAX_CACHE_SIZE', 1000)
        
        # 2. Hardware Acceleration Detection (Includes Mac M1/M2/M3)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logger.info(f"Loading models onto {self.device.upper()} for org '{org_id}'...")

        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)
        
        self.use_reranker = getattr(Config, 'USE_RERANKER', True)
        if self.use_reranker:
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device, max_length=512)
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}. Falling back.")
                self.reranker = None
        else:
            self.reranker = None

        self._pool = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    _SYNONYMS = {
        "price": "price cost rate value amount per sqft", 
        "area": "area size carpet super built sqft square feet saleable",
        "parking": "parking car park garage stilt covered two wheeler", 
        "amenities": "amenities facilities features club gym pool recreation",
        "possession": "possession handover ready completion delivery date", 
        "rera": "rera registration number approved authority",
    }

    def _normalize_expand(self, text):
        text = text.lower()
        extra = [syn for key, syn in self._SYNONYMS.items() if key in text]
        return text + " " + " ".join(extra) if extra else text

    def _get_active_entities(self):
        if not self.db.metadata: return []
        return list(set(m["source"] for m in self.db.metadata.values()))

    def _detect_query_entities(self, text):
        text_lower = text.lower()
        active_docs = self._get_active_entities()
        return [doc for doc in active_docs if doc.replace('.pdf', '').lower() in text_lower or doc.replace('-brochure', '').replace('_brochure', '').lower() in text_lower]

    def _get_embedding(self, text):
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

    def _apply_mmr(self, candidates, query_vec, k, lambda_param=0.7):
        # 3. Prevent MMR crash when candidates count is too low
        if not candidates or len(candidates) <= 1: 
            return candidates[:k]
        
        selected = []
        cand_embeds = np.vstack([self._get_embedding(c["text"]) for c in candidates])
            
        scores_q = np.dot(cand_embeds, query_vec.T).flatten()
        remaining_indices = list(range(len(candidates)))
        
        best_idx = int(np.argmax(scores_q))
        selected.append(best_idx)
        remaining_indices.remove(best_idx)
        
        while len(selected) < k and remaining_indices:
            mmr_scores = []
            for idx in remaining_indices:
                sim_q = scores_q[idx]
                sim_selected = np.max([np.dot(cand_embeds[idx], cand_embeds[s_idx]) for s_idx in selected])
                mmr_scores.append(lambda_param * sim_q - (1 - lambda_param) * sim_selected)
            
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected.append(best_idx)
            remaining_indices.remove(best_idx)
            
        return [candidates[i] for i in selected]

    def query(self, text, k=Config.DEFAULT_K):
        t_start = time.perf_counter()
        latencies = {"embedding": 0.0, "retrieval": 0.0, "reranking": 0.0, "llm": 0.0, "total": 0.0}

        if not self.ready or not text.strip(): 
            return {"results": [], "error": "Index missing.", "latency": latencies}

        k = min(max(1, k), Config.MAX_K)
        cache_key = f"{self.org_id}:{text.strip().lower()}:{k}"

        if getattr(Config, 'USE_QUERY_CACHE', True):
            with self._cache_lock:
                if cache_key in self._query_cache:
                    cached_res = self._query_cache[cache_key].copy()
                    latencies["total"] = round((time.perf_counter() - t_start) * 1000, 2)
                    cached_res["latency"] = latencies
                    cached_res["cached"] = True
                    return cached_res

        detected_entities = self._detect_query_entities(text)
        expanded = self._normalize_expand(text)
        
        t0 = time.perf_counter()
        query_vec = self._get_embedding(expanded)
        latencies["embedding"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        candidates = []
        if len(detected_entities) > 1:
            per_entity_k = max(k, 5)
            for entity in detected_entities:
                entity_hits = self.db.hybrid_search(query_vec, expanded, k=per_entity_k, allowed_source=entity)
                candidates.extend(entity_hits)
        else:
            candidates = self.db.hybrid_search(query_vec, expanded, k=k * 4)
        latencies["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        if not candidates: 
            return {"results": [], "message": "No matches found.", "latency": latencies}

        seen, unique_cands = set(), []
        for c in candidates:
            uid = (c.get("source"), c.get("page_num"), c.get("chunk_index", 0))
            if uid not in seen:
                seen.add(uid)
                unique_cands.append(c)

        t0 = time.perf_counter()
        if self.reranker and len(unique_cands) > 1:
            pairs = [(text, c["text"]) for c in unique_cands]
            logits = np.array(self.reranker.predict(pairs))
            probs = 1 / (1 + np.exp(-logits / 2.0))
            ranked = sorted(zip(probs, unique_cands), key=lambda x: x[0], reverse=True)
            
            threshold = getattr(Config, 'CALIBRATION_THRESHOLD', 0.30)
            filtered = [ {**c, "score": round(float(p), 4)} for p, c in ranked if p >= threshold ]
        else:
            filtered = unique_cands[:k]
        latencies["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        final_results = self._apply_mmr(filtered, query_vec, k)

        t0_llm = time.perf_counter()
        query_type = "comparison" if len(detected_entities) > 1 else "general"
        llm_resp = generate_answer(
            query=text, 
            chunks=final_results, 
            query_type=query_type, 
            entities=detected_entities
        )
        latencies["llm"] = round((time.perf_counter() - t0_llm) * 1000, 2)
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
            "cached":            False
        }

        if getattr(Config, 'USE_QUERY_CACHE', True):
            with self._cache_lock:
                if len(self._query_cache) >= self.MAX_CACHE_SIZE:
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[cache_key] = response

        return response

    def ingest(self, pages):
        from app.chunker import chunk_text
        t0 = time.perf_counter()
        
        with self._cache_lock:
            self._query_cache.clear()
            self._embed_cache.clear()
        
        chunks = chunk_text(pages)
        if not chunks: return {"error": "No text found.", "chunks": 0}

        embeddings = self.model.encode([c["text"] for c in chunks], batch_size=Config.BATCH_SIZE, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        if self.db.index is None: 
            self.db.build(embeddings, chunks)
        else: 
            self.db.add(embeddings, chunks)

        self.ready = True
        return {"chunks": len(chunks), "ingest_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @property
    def stats(self): 
        return {**self.db.stats, "model": Config.MODEL_NAME, "org_id": self.org_id}


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
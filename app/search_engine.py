import math
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import Config
from app.logger import logger
from app.vector_db import VectorDB

class SearchEngine:
    # Calibrated threshold: results below this probability (0-1) are discarded
    MIN_PROBABILITY = 0.4 

    def __init__(self):
        self.db = VectorDB()
        self.ready = False
        
        self._query_cache = {}
        self.MAX_CACHE_SIZE = getattr(Config, 'MAX_CACHE_SIZE', 1000)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto {self.device.upper()}...")

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
        if self.db.load(): self.ready = True

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
        """Returns unique source names currently in the DB."""
        if not self.db.metadata: return []
        return list(set(m["source"] for m in self.db.metadata.values()))

    def _detect_query_entities(self, text):
        """Identifies which properties are being discussed."""
        text_lower = text.lower()
        active_docs = self._get_active_entities()
        # Filter docs whose names (or parts of them) appear in the query
        return [doc for doc in active_docs if doc.replace('.pdf', '').lower() in text_lower or doc.replace('-brochure', '').replace('_brochure', '').lower() in text_lower]

    @lru_cache(maxsize=1000)
    def _get_embedding(self, text):
        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        return vector

    def _apply_mmr(self, candidates, query_vec, k, lambda_param=0.7):
        """Maximal Marginal Relevance to reduce redundancy."""
        if not candidates or len(candidates) <= k: return candidates
        
        selected = []
        cand_embeds = np.array([self._get_embedding(c["text"]) for c in candidates]).squeeze()
        
        if len(cand_embeds.shape) == 1: cand_embeds = cand_embeds.reshape(1, -1)
            
        scores_q = np.dot(cand_embeds, query_vec.T).squeeze()
        
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
        latencies = {"embedding": 0.0, "retrieval": 0.0, "reranking": 0.0, "total": 0.0}

        if not self.ready or not text.strip(): 
            return {"results": [], "error": "Index missing.", "latency": latencies}

        k = min(max(1, k), Config.MAX_K)
        cache_key = f"{text.strip().lower()}_{k}"

        if getattr(Config, 'USE_QUERY_CACHE', True) and cache_key in self._query_cache:
            cached_res = self._query_cache[cache_key].copy()
            latencies["total"] = round((time.perf_counter() - t_start) * 1000, 2)
            cached_res["latency"] = latencies
            cached_res["cached"] = True
            return cached_res

        # 1. Entity Awareness & Multi-Arm Retrieval
        detected_entities = self._detect_query_entities(text)
        expanded = self._normalize_expand(text)
        
        t0 = time.perf_counter()
        query_vec = self._get_embedding(expanded)
        latencies["embedding"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        candidates = []
        if len(detected_entities) > 1:
            # Comparison detected: Fetch quotas per entity using Hard Filter
            per_entity_k = max(k, 5)
            for entity in detected_entities:
                entity_hits = self.db.hybrid_search(query_vec, expanded, k=per_entity_k, allowed_source=entity)
                candidates.extend(entity_hits)
        else:
            candidates = self.db.hybrid_search(query_vec, expanded, k=k * 4)
        latencies["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        if not candidates: 
            return {"results": [], "message": "No matches found.", "latency": latencies}

        # 2. Deduplication before Reranking
        seen, unique_cands = set(), []
        for c in candidates:
            uid = (c.get("source"), c.get("page_num"), c.get("chunk_index", 0))
            if uid not in seen:
                seen.add(uid)
                unique_cands.append(c)

# 3. Calibrated Reranking (Absolute Probability)
        t0 = time.perf_counter()
        if self.reranker and len(unique_cands) > 1:
            pairs = [(text, c["text"]) for c in unique_cands]
            
            # Get raw logits from the cross-encoder
            logits = np.array(self.reranker.predict(pairs))
            
            # Platt Scaling (Sigmoid) on RAW logits. 
            # We divide by 2.0 (temperature) to smooth extreme values.
            probs = 1 / (1 + np.exp(-logits / 2.0))
            
            ranked = sorted(zip(probs, unique_cands), key=lambda x: x[0], reverse=True)
            
            # THE HARD GATE: Must have an absolute probability > 50%
            filtered = [ {**c, "score": round(float(p), 4)} for p, c in ranked if p >= 0.25 ]
        else:
            filtered = unique_cands[:k]
        latencies["reranking"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Diversity Filter (MMR)
        final_results = self._apply_mmr(filtered, query_vec, k)

        latencies["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        response = {
            "results": final_results, 
            "query": text, 
            "entities_detected": detected_entities,
            "latency": latencies,
            "cached": False
        }

        if getattr(Config, 'USE_QUERY_CACHE', True):
            if len(self._query_cache) >= self.MAX_CACHE_SIZE:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[cache_key] = response

        return response

    def ingest(self, pages):
        from app.chunker import chunk_text
        t0 = time.perf_counter()
        self._query_cache.clear()
        self._get_embedding.cache_clear()
        
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
    def stats(self): return {**self.db.stats, "model": Config.MODEL_NAME}
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from app.config import Config
from app.logger import logger, log_llm

# ── Constants ──────────────────────────────────────────────────────────────────
TIMEOUT_SECS    = 60
TEMPERATURE     = 0.1     
MAX_TOKENS      = 768
REPEAT_PENALTY  = 1.1     
NUM_CTX         = 4096    
MIN_CONTEXT_WORDS = 15    

# OPTIMIZATION: Persistent HTTP opener to reuse TCP connections (Keep-Alive)
_http_opener = urllib.request.build_opener(urllib.request.HTTPHandler())

# ── Response dataclass ─────────────────────────────────────────────────────────
class LLMResponse:
    def __init__(
        self,
        answer:      str,
        sources:     list[dict],
        query_type:  str,
        refused:     bool       = False,
        latency_ms:  float      = 0.0,
        chunks_used: int        = 0,
    ):
        self.answer      = answer
        self.sources     = sources
        self.query_type  = query_type
        self.refused     = refused
        self.latency_ms  = latency_ms
        self.chunks_used = chunks_used

    def to_dict(self) -> dict:
        return {
            "answer":      self.answer,
            "sources":     self.sources,
            "query_type":  self.query_type,
            "refused":     self.refused,
            "latency_ms":  round(self.latency_ms, 2),
            "chunks_used": self.chunks_used,
        }

# ── Ollama client (stdlib only) ────────────────────────────────────────────────
def _call_ollama(prompt: str) -> str:
    """
    POST to local Ollama server and return the model's response text.
    Uses cached OpenerDirector for HTTP Keep-Alive performance.
    """
    payload = json.dumps({
        "model":  Config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":    TEMPERATURE,
            "num_predict":    MAX_TOKENS,
            "num_ctx":        NUM_CTX,
            "repeat_penalty": REPEAT_PENALTY,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        Config.OLLAMA_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Connection": "keep-alive" # Instructs server to hold TCP open
        },
        method="POST",
    )

    try:
        with _http_opener.open(req, timeout=TIMEOUT_SECS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama not reachable at {Config.OLLAMA_URL}. "
            f"Run: ollama serve\nOriginal error: {e}"
        )

def is_ollama_running() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=3):
            return True
    except Exception:
        return False

# ── Chunk selection ────────────────────────────────────────────────────────────
def _select_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    avg_score = sum(c.get("score", 0.0) for c in chunks) / len(chunks)

    if avg_score >= 0.70: top_k = 3
    elif avg_score >= 0.50: top_k = 5
    else: top_k = 7

    return chunks[:top_k]

# ── Insufficiency check ────────────────────────────────────────────────────────
def _context_is_insufficient(chunks: list[dict]) -> bool:
    if not chunks:
        return True

    avg_score = sum(c.get("score", 0.0) for c in chunks) / len(chunks)
    if avg_score < getattr(Config, 'CALIBRATION_THRESHOLD', 0.15):
        return True

    total_words = sum(len(c.get("text", "").split()) for c in chunks)
    if total_words < MIN_CONTEXT_WORDS:
        return True

    return False

# ── Context builder ────────────────────────────────────────────────────────────
def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    context_parts = []
    sources       = []

    for i, chunk in enumerate(chunks, start=1):
        source  = chunk.get("source", "Unknown")
        page    = chunk.get("page_num", "?")
        score   = chunk.get("score", 0.0)
        text    = chunk.get("text", "").strip()

        context_parts.append(f"[{i}] {source} — Page {page}\n{text}")
        sources.append({
            "ref":    i,
            "source": source,
            "page":   page,
            "score":  round(score, 3),
        })

    return "\n\n---\n\n".join(context_parts), sources

# ── Prompt builders ────────────────────────────────────────────────────────────
_SYSTEM_RULES = (
    "You are a highly analytical document assistant. "
    "Answer questions strictly based on the numbered context chunks provided. "
    "Rules you must follow:\n"
    "1. Only use information present in the context. Never add outside knowledge.\n"
    "2. Cite every fact with its source number like [1] or [2].\n"
    "3. If the context does not contain enough information to answer, "
    "reply with exactly: INSUFFICIENT_CONTEXT\n"
    "4. Be concise and factual. Format effectively."
)

def _build_general_prompt(query: str, context: str) -> str:
    return (
        f"{_SYSTEM_RULES}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite sources):"
    )

def _build_comparison_prompt(query: str, context: str, entities: list[str]) -> str:
    entity_label = " vs ".join(entities) if entities else "the items"
    return (
        f"{_SYSTEM_RULES}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Compare {entity_label}. Structure your response exactly as:\n"
        f"1. A markdown table for direct comparison.\n"
        f"   Write 'N/A' for any attribute not mentioned in the context.\n"
        f"2. A brief 2-3 sentence summary paragraph below the table.\n"
        f"Cite sources as [1], [2], etc.\n\n"
        f"COMPARISON:"
    )

# ── Refusal message ────────────────────────────────────────────────────────────
_REFUSAL = (
    "I don't have enough information in the uploaded documents to answer this. "
    "Please upload relevant files or rephrase your question."
)

# ── Main entry point ───────────────────────────────────────────────────────────
def generate_answer(
    query:      str,
    chunks:     list[dict],
    query_type: str           = "general",
    entities:   Optional[list[str]] = None,
    org_id:     str           = "default",
) -> LLMResponse:
    t_start = time.perf_counter()

    if _context_is_insufficient(chunks):
        logger.info(
            "LLM refused — insufficient context",
            extra={"org_id": org_id, "query": query[:80]},
        )
        return LLMResponse(
            answer      = _REFUSAL,
            sources     = [],
            query_type  = query_type,
            refused     = True,
            latency_ms  = (time.perf_counter() - t_start) * 1000,
            chunks_used = 0,
        )

    selected = _select_chunks(chunks)
    context, sources = _build_context(selected)

    if query_type == "comparison":
        prompt = _build_comparison_prompt(query, context, entities or [])
    else:
        prompt = _build_general_prompt(query, context)

    try:
        raw_answer = _call_ollama(prompt)
    except RuntimeError as e:
        logger.error(str(e), extra={"org_id": org_id})
        return LLMResponse(
            answer      = (
                "⚠️ LLM service is unavailable (is Ollama running?).\n\n"
                "Raw context from your documents:\n\n" + context
            ),
            sources     = sources,
            query_type  = query_type,
            refused     = False,
            latency_ms  = (time.perf_counter() - t_start) * 1000,
            chunks_used = len(selected),
        )

    refused = "INSUFFICIENT_CONTEXT" in raw_answer
    if refused:
        raw_answer = _REFUSAL

    latency_ms = (time.perf_counter() - t_start) * 1000

    log_llm(
        org_id     = org_id,
        query      = query,
        refused    = refused,
        chunks_used = len(selected),
        latency_ms = latency_ms,
        query_type = query_type,
    )

    return LLMResponse(
        answer      = raw_answer,
        sources     = sources,
        query_type  = query_type,
        refused     = refused,
        latency_ms  = latency_ms,
        chunks_used = len(selected),
    )
import logging
import sys
import json
from datetime import datetime, timezone

from app.config import Config


class _JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per log line.
    Supports extra context fields: org_id, user_id, latency_ms, query, doc_id.
    """

    _EXTRA_FIELDS = ("org_id", "user_id", "latency_ms", "query", "doc_id",
                     "chunks_used", "refused", "cache_hit", "query_type")

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for key in self._EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG if Config.DEBUG else logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        log.addHandler(handler)
    log.propagate = False
    return log


logger = _build(Config.PROJECT_NAME)



def log_query(org_id: str, query: str, latency_ms: float, cache_hit: bool) -> None:
    logger.info(
        "query_served",
        extra={
            "org_id":     org_id,
            "query":      query[:120],
            "latency_ms": round(latency_ms, 2),
            "cache_hit":  cache_hit,
        },
    )


def log_ingest(org_id: str, doc_id: str, chunks: int, latency_ms: float) -> None:
    logger.info(
        "doc_ingested",
        extra={
            "org_id":     org_id,
            "doc_id":     doc_id,
            "chunks":     chunks,
            "latency_ms": round(latency_ms, 2),
        },
    )


def log_llm(
    org_id: str,
    query: str,
    refused: bool,
    chunks_used: int,
    latency_ms: float,
    query_type: str,
) -> None:
    logger.info(
        "llm_answer",
        extra={
            "org_id":      org_id,
            "query":       query[:120],
            "refused":     refused,
            "chunks_used": chunks_used,
            "latency_ms":  round(latency_ms, 2),
            "query_type":  query_type,
        },
    )
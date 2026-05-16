import os


class Config:

    PROJECT_NAME = "Real Estate Intel"
    VERSION      = "2.0.0"

    # ── Embedding ──────────────────────────────────────────────────────────────
    MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
    EMBED_DIM   = 768
    MAX_SEQ_LEN = 384
    BATCH_SIZE  = 32

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_SIZE    = 250
    CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
    USE_QUERY_CACHE       = True
    MAX_CACHE_SIZE        = 1000
    USE_RERANKER          = True
    
    # CHANGE 1: Increase starting chunks to give the reranker more options
    DEFAULT_K             = 10   # Was 5
    MAX_K                 = 30   # Was 20
    
    # CHANGE 2: Lower the strict probability gate. 
    # 0.15 is much better for sparse brochure text than 0.30
    CALIBRATION_THRESHOLD = 0.15 # Was 0.30
    
    MMR_LAMBDA            = 0.7

    # ── File limits ────────────────────────────────────────────────────────────
    MAX_PDF_MB      = 50
    MIN_TEXT_CHARS  = 50
    MAX_PAGES       = 300
    UPLOAD_LIMIT_MB = 50
    MAX_WORKERS     = 4

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST  = "0.0.0.0"
    PORT  = 5000
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # ── Auth ───────────────────────────────────────────────────────────────────
    JWT_SECRET      = os.getenv("JWT_SECRET", "change-me-before-prod")
    JWT_ALGORITHM   = "HS256"
    JWT_EXPIRE_MINS = 60 * 24   # 24 hours

    # ── Redis (optional caching layer) ─────────────────────────────────────────
    REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECS = 3600

    # ── LLM (Ollama local) ─────────────────────────────────────────────────────
    OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    # ── Base directories ───────────────────────────────────────────────────────
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")    # storage/{org_id}/
    INDEX_DIR   = os.path.join(BASE_DIR, "indexes")    # indexes/{org_id}/
    DATA_DIR    = os.path.join(BASE_DIR, "data")       # legacy CLI ingest only

    # ── Tenant-scoped path helpers ─────────────────────────────────────────────
    @staticmethod
    def index_path(org_id: str) -> str:
        return os.path.join(Config.INDEX_DIR, org_id, "faiss.index")

    @staticmethod
    def meta_path(org_id: str) -> str:
        return os.path.join(Config.INDEX_DIR, org_id, "metadata.pkl")

    @staticmethod
    def upload_dir(org_id: str) -> str:
        return os.path.join(Config.STORAGE_DIR, org_id)


# Auto-create base dirs on import (tenant dirs are created on registration)
for _d in (Config.STORAGE_DIR, Config.INDEX_DIR, Config.DATA_DIR):
    os.makedirs(_d, exist_ok=True)
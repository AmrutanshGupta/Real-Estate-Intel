import os

class Config:
    PROJECT_NAME = "Real Estate Intel"
    VERSION      = "1.0.0"

    # Embedding Model
    MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
    EMBED_DIM   = 768
    MAX_SEQ_LEN = 384

    # Chunking limits
    CHUNK_SIZE    = 250
    CHUNK_OVERLAP = 50
    BATCH_SIZE    = 32  # Reduced to maintain stable VRAM usage

    # Caching & Optimization Toggles
    USE_QUERY_CACHE = True
    MAX_CACHE_SIZE  = 1000
    USE_RERANKER    = True  # Toggle to False if end-to-end latency > 50% target

    MAX_PDF_MB     = 50
    MIN_TEXT_CHARS = 50
    MAX_PAGES      = 300

    DEFAULT_K   = 5
    MAX_K       = 20
    HOST        = "0.0.0.0"
    PORT        = 5000
    DEBUG       = os.getenv("DEBUG", "false").lower() == "true"
    UPLOAD_LIMIT_MB = 50
    MAX_WORKERS = 4

    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR    = os.path.join(BASE_DIR, "data")
    INDEX_DIR   = os.path.join(BASE_DIR, "faiss_index")
    INDEX_PATH  = os.path.join(INDEX_DIR, "index.faiss")
    META_PATH   = os.path.join(INDEX_DIR, "metadata.pkl")
    UPLOAD_DIR  = os.path.join(DATA_DIR, "uploads")

for _d in (Config.DATA_DIR, Config.INDEX_DIR, Config.UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)
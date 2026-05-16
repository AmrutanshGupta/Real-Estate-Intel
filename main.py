import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("OMP_NUM_THREADS", "1")

import jwt
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.config import Config
from app.logger import logger
from app.pdf_loader import load_document
from app.search_engine import get_engine, invalidate_engine
from app.vector_db import invalidate_db


# ── Security constants ─────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"you are now",
    r"disregard (your|the) (system|previous)",
    r"forget (everything|all|your) (above|prior|previous)",
    r"act as (a |an )(?!real estate)",
    r"jailbreak",
]

_bearer = HTTPBearer(auto_error=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)


def create_token(org_id: str) -> str:
    import datetime
    payload = {
        "org_id": org_id,
        "sub":    org_id,
        "exp":    datetime.datetime.utcnow()
                  + datetime.timedelta(minutes=Config.JWT_EXPIRE_MINS),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


# ── Auth dependency ────────────────────────────────────────────────────────────

def get_org_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    x_api_key:   str | None = Header(None),
) -> str:
    """
    Resolves org_id from:
      1. Bearer JWT token  (Authorization: Bearer <token>)
      2. X-Api-Key header  (X-Api-Key: <token>)
      3. DEBUG fallback    (org_id = "default", only when DEBUG=true)
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif x_api_key:
        token = x_api_key

    if not token:
        if Config.DEBUG:
            return "default"
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        payload = jwt.decode(
            token,
            Config.JWT_SECRET,
            algorithms=[Config.JWT_ALGORITHM],
        )
        return payload.get("org_id") or payload.get("sub", "default")

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {Config.PROJECT_NAME} v{Config.VERSION}")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=Config.PROJECT_NAME,
    version=Config.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    org_id: str
    secret: str   # validate against hashed secret in DB for production


class SearchQuery(BaseModel):
    query: str
    k:     int = Config.DEFAULT_K


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest):
    """
    Issues a JWT for the given org_id.
    TODO: validate req.secret against a real database before production.
    """
    if not req.org_id or len(req.org_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid org_id.")

    # Create the org's storage and index directories on first login
    os.makedirs(Config.upload_dir(req.org_id), exist_ok=True)
    os.makedirs(os.path.join(Config.INDEX_DIR, req.org_id), exist_ok=True)

    token = create_token(req.org_id)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "org_id":       req.org_id,
        "expires_in":   Config.JWT_EXPIRE_MINS * 60,
    }


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
async def health():
    return {"status": "ok", "version": Config.VERSION}


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["documents"])
async def upload(
    files:  list[UploadFile] = File(...),
    org_id: str              = Depends(get_org_id),
):
    """
    Upload and index PDF or DOCX files for the authenticated tenant.
    Files are stored at storage/{org_id}/{filename}.
    Ingestion is synchronous here — swap engine.ingest() for a Celery
    task call when you need fully async processing.
    """
    upload_dir = Path(Config.upload_dir(org_id))
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    engine  = get_engine(org_id)

    for f in files:
        ext = Path(f.filename).suffix.lower()

        if ext not in (".pdf", ".docx", ".doc"):
            results.append({"file": f.filename, "error": "Only PDF and DOCX are accepted."})
            continue

        size_bytes = 0
        dest       = upload_dir / f.filename

        try:
            content    = await f.read()
            size_bytes = len(content)

            if size_bytes > Config.UPLOAD_LIMIT_MB * 1024 * 1024:
                results.append({
                    "file":  f.filename,
                    "error": f"File exceeds {Config.UPLOAD_LIMIT_MB} MB limit.",
                })
                continue

            dest.write_bytes(content)

        except Exception as e:
            logger.error(f"Save failed for {f.filename}: {e}", extra={"org_id": org_id})
            results.append({"file": f.filename, "error": "File save failed."})
            continue

        doc_id = Path(f.filename).stem
        pages  = load_document(str(dest), org_id=org_id, doc_id=doc_id)

        if not pages:
            results.append({"file": f.filename, "error": "No extractable text found."})
            dest.unlink(missing_ok=True)
            continue

        stats = engine.ingest(pages)
        invalidate_db(org_id)   # force registry reload after index update

        results.append({
            "file":      f.filename,
            "doc_id":    doc_id,
            "pages":     len(pages),
            "chunks":    stats.get("chunks", 0),
            "ingest_ms": stats.get("ingest_ms", 0),
            "org_id":    org_id,
        })

        logger.info(
            f"Uploaded and indexed '{f.filename}'",
            extra={"org_id": org_id, "doc_id": doc_id},
        )

    return {"files": results, "index": engine.stats}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/api/search", tags=["search"])
async def search(
    req:    SearchQuery,
    org_id: str = Depends(get_org_id),
):
    """
    Tenant-scoped RAG query.
    Returns LLM-generated answer + raw retrieved chunks + latency breakdown.
    """
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query exceeds 500 characters.")

    # Prompt injection guard
    if detect_injection(query):
        logger.warning(
            "Injection attempt blocked",
            extra={"org_id": org_id, "query": query[:80]},
        )
        raise HTTPException(
            status_code=400,
            detail="Query contains disallowed instructions.",
        )

    engine = get_engine(org_id)
    result = engine.query(query, req.k)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


# ── Documents ──────────────────────────────────────────────────────────────────

@app.get("/api/documents", tags=["documents"])
async def documents(org_id: str = Depends(get_org_id)):
    """List all indexed documents for the authenticated tenant."""
    engine = get_engine(org_id)

    if not engine.ready:
        return {"documents": [], "org_id": org_id}

    doc_stats: dict[str, dict] = {}
    for meta in engine.db.metadata.values():
        src = meta["source"]
        if src not in doc_stats:
            doc_stats[src] = {"name": src, "chunks": 0, "pages": set()}
        doc_stats[src]["chunks"] += 1
        doc_stats[src]["pages"].add(meta["page_num"])

    docs = [
        {
            "name":   d["name"],
            "chunks": d["chunks"],
            "pages":  len(d["pages"]),
        }
        for d in doc_stats.values()
    ]
    return {
        "documents": sorted(docs, key=lambda x: x["name"]),
        "org_id":    org_id,
    }


@app.delete("/api/documents/{name}", tags=["documents"])
async def delete_document(
    name:   str,
    org_id: str = Depends(get_org_id),
):
    """
    Remove a document from the tenant's index and delete its file from storage.
    Rebuilds FAISS index from remaining vectors.
    """
    import numpy as np
    import faiss as _faiss

    engine = get_engine(org_id)

    if not engine.ready:
        raise HTTPException(status_code=503, detail="No index loaded.")

    old_meta = engine.db.metadata
    keep_ids = [i for i, m in old_meta.items() if m["source"] != name]

    if len(keep_ids) == len(old_meta):
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")

    try:
        if keep_ids:
            cpu_index = (
                _faiss.index_gpu_to_cpu(engine.db.index)
                if engine.db.res
                else engine.db.index
            )
            all_vecs  = cpu_index.reconstruct_n(0, cpu_index.ntotal)
            kept_vecs = np.array([all_vecs[i] for i in keep_ids], dtype="float32")
            kept_meta = [old_meta[i] for i in keep_ids]
            engine.db.build(kept_vecs, kept_meta)
        else:
            engine.db.index    = None
            engine.db.metadata = {}
            engine.db._persist()
            engine.ready = False

    except Exception as e:
        logger.error(f"Delete rebuild failed: {e}", extra={"org_id": org_id})
        raise HTTPException(status_code=500, detail="Index rebuild failed.")

    # Remove the stored file
    file_path = Path(Config.upload_dir(org_id)) / name
    file_path.unlink(missing_ok=True)

    # Evict caches so next request gets fresh state
    invalidate_engine(org_id)
    invalidate_db(org_id)

    logger.info(f"Deleted '{name}'", extra={"org_id": org_id})
    return {
        "deleted": name,
        "org_id":  org_id,
        "index":   engine.stats,
    }


# ── Stats ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats", tags=["system"])
async def stats(org_id: str = Depends(get_org_id)):
    return {**get_engine(org_id).stats, "org_id": org_id}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
    )
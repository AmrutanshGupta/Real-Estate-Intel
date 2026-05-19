import os
import re
import time
import shutil
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Config
from app.logger import logger
from app.pdf_loader import load_document
from app.search_engine import get_engine, invalidate_engine
from app.vector_db import invalidate_db

# ── New Production Architecture Imports ────────────────────────────────────────
from app.db import init_db, history_collection
from app.auth import auth_router, get_current_tenant


# ── Security constants ─────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"you are now",
    r"disregard (your|the) (system|previous)",
    r"forget (everything|all|your) (above|prior|previous)",
    r"act as (a |an )(?!real estate)",
    r"jailbreak",
]

def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {Config.PROJECT_NAME} v{Config.VERSION}")
    
    # Initialize MongoDB Collections & Indexes
    init_db()
    
    # Verify local LLM accessibility
    from app.llm_layer import is_ollama_running
    if not is_ollama_running():
        logger.warning(
            f"Ollama server not detected! "
            f"LLM generation will fallback to raw context. "
            f"Ensure `ollama serve` is running."
        )
        
    yield
    logger.info("Shutting down.")


# ── App Initialization ─────────────────────────────────────────────────────────

app = FastAPI(
    title=Config.PROJECT_NAME,
    version=Config.VERSION,
    lifespan=lifespan,
)

# Configured for Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach external routers (handles /api/auth/login and /api/auth/oauth/callback)
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str
    k:     int = Config.DEFAULT_K


# ── Health & Stats ─────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
async def health():
    return {"status": "ok", "version": Config.VERSION}

@app.get("/api/stats", tags=["system"])
async def stats(org_id: str = Depends(get_current_tenant)):
    return {**get_engine(org_id).stats, "org_id": org_id}


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["documents"])
async def upload(
    files:  list[UploadFile] = File(...),
    org_id: str              = Depends(get_current_tenant),
):
    """
    Upload and index PDF or DOCX files for the authenticated tenant.
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
        
        # ── CRITICAL FIX 1: Offload CPU-heavy PDF loading to background thread ──
        pages = await asyncio.to_thread(load_document, str(dest), org_id=org_id, doc_id=doc_id)

        if not pages:
            results.append({"file": f.filename, "error": "No extractable text found."})
            dest.unlink(missing_ok=True)
            continue

        # ── CRITICAL FIX 2: Await the new async ingest method ──
        stats = await engine.ingest(pages)
        invalidate_db(org_id)   # force registry reload after index update

        if stats.get("chunks", 0) == 0:
            results.append({"file": f.filename, "error": "Text extracted, but no valid chunks generated."})
            dest.unlink(missing_ok=True)
            continue
        # ───────────────────────────────────────────────────────────────────

        results.append({
            "file":      f.filename,
            "doc_id":    doc_id,
            "pages":     len(pages),
            "chunks":    stats.get("chunks", 0),
            "ingest_ms": stats.get("ingest_ms", 0),
            "org_id":    org_id,
        })

        logger.info(f"Uploaded and indexed '{f.filename}'", extra={"org_id": org_id, "doc_id": doc_id})

    return {"files": results, "index": engine.stats}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/api/search", tags=["search"])
async def search(
    req:    SearchQuery,
    org_id: str = Depends(get_current_tenant),
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
        logger.warning("Injection attempt blocked", extra={"org_id": org_id, "query": query[:80]})
        raise HTTPException(status_code=400, detail="Query contains disallowed instructions.")

    engine = get_engine(org_id)
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No documents indexed for this organization yet.")

    # ── CRITICAL FIX 3: Await the async query method ──
    result = await engine.query(query, req.k)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    # Log query to MongoDB for Dashboard History
    if history_collection is not None:
        history_collection.insert_one({
            "org_id": org_id, 
            "query": query, 
            "results": len(result.get("results", []))
        })

    return result


# ── Documents ──────────────────────────────────────────────────────────────────

@app.get("/api/documents", tags=["documents"])
async def documents(org_id: str = Depends(get_current_tenant)):
    """List all indexed documents for the authenticated tenant."""
    engine = get_engine(org_id)

    if not engine.ready or not engine.db.metadata:
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
    org_id: str = Depends(get_current_tenant),
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
            
            # Offload heavy FAISS operations to thread
            await asyncio.to_thread(engine.db.build, kept_vecs, kept_meta)
        else:
            engine.db.index    = None
            engine.db.metadata = {}
            await asyncio.to_thread(engine.db._persist)
            engine.ready = False

    except Exception as e:
        logger.error(f"Delete rebuild failed: {e}", extra={"org_id": org_id})
        raise HTTPException(status_code=500, detail="Index rebuild failed.")

    # Remove the stored file
    file_path = Path(Config.upload_dir(org_id)) / name
    await asyncio.to_thread(file_path.unlink, missing_ok=True)

    # Evict caches so next request gets fresh state
    invalidate_engine(org_id)
    invalidate_db(org_id)

    logger.info(f"Deleted '{name}'", extra={"org_id": org_id})
    return {
        "deleted": name,
        "org_id":  org_id,
        "index":   engine.stats,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
    )
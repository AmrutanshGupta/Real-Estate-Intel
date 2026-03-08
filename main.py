import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Config
from app.logger import logger
from app.pdf_loader import load_pdf
from app.search_engine import SearchEngine

engine = SearchEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if engine.db.index:
        engine.db._persist()

app = FastAPI(title=Config.PROJECT_NAME, version=Config.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    k: int = Config.DEFAULT_K

@app.get("/api/health")
async def health():
    db_healthy = engine.db.index is not None and engine.db.index.ntotal >= 0
    status = "ok" if (engine.ready and db_healthy) else "degraded"
    if status != "ok":
        raise HTTPException(status_code=503, detail="Index not ready")
    return {"status": status, "version": Config.VERSION, **engine.stats}

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        if not f.filename.lower().endswith('.pdf'):
            results.append({"file": f.filename, "error": "Only PDFs allowed."})
            continue

        dest = Path(Config.UPLOAD_DIR) / f.filename
        try:
            content = await f.read()
            dest.write_bytes(content)
        except Exception as e:
            logger.error(f"Save failed: {e}")
            continue

        pages = load_pdf(str(dest))
        if not pages:
            results.append({"file": f.filename, "error": "No extractable text."})
            dest.unlink(missing_ok=True)
            continue

        stats = engine.ingest(pages)
        results.append({
            "file": f.filename,
            "pages": len(pages),
            "chunks": stats.get("chunks", 0),
            "ingest_ms": stats.get("ingest_ms", 0),
        })

    return {"files": results, "index": engine.stats}

@app.post("/api/search")
async def search(req: SearchQuery):
    if len(req.query) > 500:
        raise HTTPException(status_code=400, detail="Query exceeds 500 characters.")

    retrieval_res = engine.query(req.query, req.k)
    
    if "error" in retrieval_res:
        raise HTTPException(status_code=503, detail=retrieval_res["error"])

    return retrieval_res

@app.get("/api/documents")
async def documents():
    if not engine.ready:
        return {"documents": []}

    doc_stats = {}
    for meta in engine.db.metadata.values():
        src = meta["source"]
        if src not in doc_stats:
            doc_stats[src] = {"name": src, "chunks": 0, "pages": set()}
        doc_stats[src]["chunks"] += 1
        doc_stats[src]["pages"].add(meta["page_num"])

    docs = [{"name": d["name"], "chunks": d["chunks"], "pages": len(d["pages"])} for d in doc_stats.values()]
    return {"documents": sorted(docs, key=lambda x: x["name"])}

@app.delete("/api/documents/{name}")
async def delete_document(name: str):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="No index loaded.")

    old_meta = engine.db.metadata
    keep_ids = [i for i, m in old_meta.items() if m["source"] != name]

    if len(keep_ids) == len(old_meta):
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")

    import faiss
    import numpy as np

    try:
        if len(keep_ids) > 0:
            cpu_index = faiss.index_gpu_to_cpu(engine.db.index) if engine.db.res else engine.db.index
            all_vecs = cpu_index.reconstruct_n(0, cpu_index.ntotal)
            kept_vecs = np.array([all_vecs[i] for i in keep_ids], dtype="float32")
            kept_meta = [old_meta[i] for i in keep_ids]
            
            engine.db.build(kept_vecs, kept_meta)
        else:
            engine.db.index = None
            engine.db.metadata = {}
            engine.db._persist()
            engine.ready = False
            
        return {"deleted": name, "index": engine.stats}
    except Exception as e:
        logger.error(f"Failed to delete {name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanly delete document.")

@app.get("/api/stats")
async def stats():
    return engine.stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=Config.HOST, port=Config.PORT, reload=Config.DEBUG)
import os
import time
from pathlib import Path

# Keep CPU threads minimal since we are offloading to the GPU
os.environ.setdefault("OMP_NUM_THREADS", "1")

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from app.config import Config
from app.logger import logger
from app.pdf_loader import load_pdf
from app.search_engine import SearchEngine

app = Flask(__name__)
# Hard limit at the WSGI layer. Drops the connection if they send >50MB.
app.config["MAX_CONTENT_LENGTH"] = Config.UPLOAD_LIMIT_MB * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": "*"}})

engine = SearchEngine()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _allowed(filename):
    return '.' in filename and filename.lower().rsplit('.', 1)[1] == 'pdf'

def _error(msg, status=400):
    return jsonify({"error": msg}), status

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return _error(f"Payload exceeds the {Config.UPLOAD_LIMIT_MB}MB absolute limit.", 413)

@app.errorhandler(Exception)
def handle_internal_error(e):
    logger.error(f"Unhandled Exception: {str(e)}")
    return _error("Internal server error. The engineering team has been notified (probably).", 500)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    # A real health check verifies the components, it doesn't just return True.
    db_healthy = engine.db.index is not None and engine.db.index.ntotal >= 0
    status = "ok" if (engine.ready and db_healthy) else "degraded"
    
    return jsonify({
        "status": status,
        "version": Config.VERSION,
        **engine.stats,
    }), 200 if status == "ok" else 503

@app.post("/api/upload")
def upload():
    if "files" not in request.files:
        return _error("No files in request.")

    files = request.files.getlist("files")
    if not files or all(f.filename == '' for f in files):
        return _error("Empty file payload.")

    results = []
    for f in files:
        if not _allowed(f.filename):
            results.append({"file": f.filename, "error": "Not a PDF. Rejected."})
            continue

        name = secure_filename(f.filename)
        dest = Path(Config.UPLOAD_DIR) / name
        
        try:
            f.save(dest)
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}")
            results.append({"file": name, "error": "Disk write failure."})
            continue

        pages = load_pdf(dest)
        if not pages:
            results.append({"file": name, "error": "No extractable text found. Scanned PDF or corrupted."})
            # Clean up the useless file
            dest.unlink(missing_ok=True)
            continue

        try:
            stats = engine.ingest(pages)
            results.append({
                "file": name,
                "pages": len(pages),
                "chunks": stats.get("chunks", 0),
                "ingest_ms": stats.get("ingest_ms", 0),
            })
        except Exception as e:
            logger.error(f"Ingestion failed for {name}: {e}")
            results.append({"file": name, "error": "ML pipeline failed during ingestion."})

    return jsonify({"files": results, "index": engine.stats})

@app.post("/api/search")
def search():
    body = request.get_json(silent=True) or {}
    q = (body.get("query") or "").strip()
    
    try:
        k = int(body.get("k", Config.DEFAULT_K))
    except ValueError:
        return _error("Parameter 'k' must be an integer.")

    if not q:
        return _error("'query' is required.")
    
    # Input length validation to prevent malicious token flooding
    if len(q) > 500:
        return _error("Query too long. Keep it under 500 characters.")

    try:
        result = engine.query(q, k)
    except Exception as e:
        logger.error(f"Search failure for query '{q[:20]}': {e}")
        return _error("Search engine encountered a fatal error.", 500)

    if "error" in result and not result.get("results"):
        return _error(result["error"], 503)

    return jsonify(result)

@app.get("/api/documents")
def documents():
    if not engine.ready:
        return jsonify({"documents": []})

    doc_stats = {}
    for meta in engine.db.metadata.values():
        src = meta["source"]
        if src not in doc_stats:
            doc_stats[src] = {"name": src, "chunks": 0, "pages": set()}
        doc_stats[src]["chunks"] += 1
        doc_stats[src]["pages"].add(meta["page_num"])

    docs = [{ "name": d["name"], "chunks": d["chunks"], "pages": len(d["pages"]) } for d in doc_stats.values()]
    return jsonify({"documents": sorted(docs, key=lambda x: x["name"])})

@app.delete("/api/documents/<name>")
def delete_document(name):
    if not engine.ready:
        return _error("No index loaded.", 503)

    old_meta = engine.db.metadata
    keep_ids = [i for i, m in old_meta.items() if m["source"] != name]

    if len(keep_ids) == len(old_meta):
        return _error(f"Document '{name}' not found.", 404)

    import faiss
    import numpy as np

    try:
        if len(keep_ids) > 0:
            # Reconstruct needs to happen on CPU usually, depending on faiss-gpu version
            cpu_index = faiss.index_gpu_to_cpu(engine.db.index) if engine.db.res else engine.db.index
            all_vecs = cpu_index.reconstruct_n(0, cpu_index.ntotal)
            kept_vecs = np.array([all_vecs[i] for i in keep_ids], dtype="float32")
            kept_meta = [old_meta[i] for i in keep_ids]
            
            # Rebuild the index completely
            engine.db.build(kept_vecs, kept_meta)
        else:
            engine.db.index = None
            engine.db.metadata = {}
            engine.db._persist()
            engine.ready = False
            
        return jsonify({"deleted": name, "index": engine.stats})
    except Exception as e:
        logger.error(f"Failed to delete {name}: {e}")
        return _error("Failed to cleanly delete document.", 500)

@app.get("/api/stats")
def stats():
    return jsonify(engine.stats)

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
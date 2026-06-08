import os
import shutil
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Config
from app.auth import auth_router, get_current_tenant
from app.dashboard import admin_router
from app.search_engine import get_engine
from app.pdf_loader import load_document

app = FastAPI(title=Config.PROJECT_NAME, version=Config.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


class SearchRequest(BaseModel):
    query: str
    k: int = Config.DEFAULT_K

@app.post("/api/search")
def execute_search(req: SearchRequest, org_id: str = Depends(get_current_tenant)):
    engine = get_engine(org_id)
    
    if not engine.ready:
        raise HTTPException(status_code=400, detail="No documents indexed for this organization yet.")
        
    response = engine.query(req.query, k=req.k)
    return response


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), org_id: str = Depends(get_current_tenant)):
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
        
    tenant_upload_dir = Config.upload_dir(org_id)
    os.makedirs(tenant_upload_dir, exist_ok=True)
    
    file_path = os.path.join(tenant_upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pages = load_document(file_path, org_id=org_id)
        if not pages:
            raise HTTPException(status_code=422, detail="Could not extract readable text from document.")
            
        engine = get_engine(org_id)
        result = engine.ingest(pages)
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_processed": result.get("chunks", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@app.on_event("startup")
def startup_event():
    from app.llm_layer import is_ollama_running
    from app.logger import logger
    
    if not is_ollama_running():
        logger.warning(
            "Ollama server not detected! "
            "LLM generation will fallback to returning raw context. "
            f"Please ensure `ollama serve` is running and `{Config.OLLAMA_MODEL}` is pulled."
        )
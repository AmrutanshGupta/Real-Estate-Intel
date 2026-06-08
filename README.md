# Real Estate Intel

A self-hosted, multi-tenant RAG platform for querying real estate documents in natural language. Upload property brochures or any PDF/DOCX, ask questions, and get grounded, cited answers from a locally running LLM — no external API calls.

---

## How It Works

Each organisation gets an isolated document index. Files are parsed, semantically chunked, embedded, and stored in a hybrid FAISS + BM25 index. Queries go through embedding, hybrid retrieval, cross-encoder reranking, MMR diversity filtering, and finally Ollama for answer generation.

---

## System Design

```
       Upload pipeline                        Query pipeline
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │      POST /api/upload    │          │      POST /api/search    │
  │  JWT-auth · PDF or DOCX  │          │   JWT-auth · plain text  │
  └─────────────┬────────────┘          └─────────────┬────────────┘
                │                                     │
                ▼                                     ▼
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │       pdf_loader.py      │          │     Synonym expansion    │
  │  PyMuPDF · readability   │          │  price·area·parking·RERA │
  └─────────────┬────────────┘          └─────────────┬────────────┘
                │                                     │
                ▼                                     ▼
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │        chunker.py        │          │       Embed query        │
  │  MiniLM cosine · 0.45    │          │  all-mpnet-base-v2 · LRU │
  └─────────────┬────────────┘          └─────────────┬────────────┘
                │                                     │
                ▼                                     ▼
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │      Embed chunks        │          │     Entity detection     │
  │  all-mpnet-base-v2·batch │          │  single vs multi-doc     │
  └─────────────┬────────────┘          └─────────────┬────────────┘
                │                                     │
                └──────────────┬──────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐     ┌──────────────┐
              │           vector_db.py         │- - -│   auth.py    │
              │  FAISS IndexFlatIP + BM25      │     │  JWT · OAuth │
              │  RRF fusion · per-tenant       │     └──────────────┘
              └────────────────┬───────────────┘
                               │                     ┌──────────────┐
                               ▼                     │   MongoDB    │
              ┌────────────────────────────────┐- - -│  tenants·    │
              │     Cross-encoder reranker     │     │  history     │
              │  ms-marco-MiniLM · sigmoid≥0.30│     └──────────────┘
              │  entity consistency gate       │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       MMR diversity filter     │
              │   balance relevance·coverage   │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      llm_layer.py — Ollama     │
              │  qwen2.5:1.5b · cited answer   │
              │  INSUFFICIENT_CONTEXT fallback │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │          JSON response         │
              │  answer · sources [1][2]       │
              │  latency breakdown · refused   │
              └────────────────────────────────┘

  Legend:  [Upload path]  [Query path]  [Shared layer]  [Support services - - -]
```

---

## Architecture

### Tenant Isolation

Every request is scoped to an `org_id` from the JWT. Each tenant gets separate upload storage, FAISS index, BM25 index, and query/embedding caches. No data crosses tenant boundaries at any layer.

### Ingestion Pipeline

1. **Loading** — PyMuPDF extracts PDF text page-by-page with a readability score fallback for scanned content. DOCX is handled by `python-docx` in ~40-paragraph blocks.
2. **Chunking** — Every sentence is encoded with `all-MiniLM-L6-v2`. Adjacent cosine similarities are computed; a new chunk starts when similarity drops below 0.45 (topic boundary). Produces variable-length, semantically coherent chunks.
3. **Indexing** — Chunks are batch-encoded with `all-mpnet-base-v2`, L2-normalised, and added to a FAISS `IndexFlatIP` alongside a BM25 index. Both are persisted to disk.

### Query Pipeline

1. **Expansion** — Domain synonyms are injected for terms like price, area, parking, amenities, possession, and RERA.
2. **Retrieval** — FAISS dense search and BM25 keyword search run in parallel. Results are fused via Reciprocal Rank Fusion (RRF, k=60). For multi-property queries, retrieval runs per entity and results are merged.
3. **Reranking** — The cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair. Scores are sigmoid-scaled (temperature=2.0); chunks below 0.30 are dropped. An entity-consistency gate penalises (0.25×) any chunk missing the query's proper-noun tokens.
4. **Diversity** — MMR deduplication balances relevance with coverage across the surviving chunks.
5. **Generation** — 3–7 chunks are selected based on average score, assembled into a numbered context block, and sent to Ollama. The model must cite every fact as `[1]`, `[2]` or return `INSUFFICIENT_CONTEXT`.

### Performance Optimisations

- INT8 dynamic quantization applied to both the embedding model and reranker on CPU
- LRU embedding and query caches (thread-safe, configurable size)
- Persistent HTTP Keep-Alive connection to Ollama
- All blocking operations offloaded via `asyncio.to_thread`

---

## Key Modules

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app, CORS, startup Ollama check |
| `pdf_loader.py` | PDF/DOCX extraction with readability scoring and text cleaning |
| `chunker.py` | Semantic sentence-boundary chunking using cosine similarity |
| `vector_db.py` | Per-tenant FAISS + BM25 index, hybrid search, RRF fusion, persistence |
| `search_engine.py` | Full query and ingest orchestration, model loading, caching, reranking, MMR |
| `llm_layer.py` | Ollama client, prompt construction, chunk selection, refusal handling |
| `auth.py` | JWT issue/verify, standard login, Google/GitHub OAuth callback |
| `config.py` | Central config with env-var overrides |
| `logger.py` | Structured JSON logging with per-request context fields |
| `database.py` | MongoDB connection pool, tenant collection, query history |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Model to use for generation |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `MODEL_NAME` | `sentence-transformers/all-mpnet-base-v2` | Embedding model |
| `DEFAULT_K` | `10` | Chunks fetched per query |
| `USE_RERANKER` | `False` | Enable cross-encoder reranking |
| `MAX_PDF_MB` | `50` | Max upload file size |
| `JWT_SECRET` | env `JWT_SECRET` | Change before production |
| `MONGO_URI` | env `MONGO_URI` | MongoDB connection string |

---

## Setup — Windows (Git Bash)

Follow these steps exactly, one command at a time.

### Step 1 — Create and activate virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This takes 3–10 minutes the first time. It downloads PyTorch, FAISS, etc.

### Step 3 — Add PDF files

Copy PDF files into the `data/` folder inside the project. You can download sample real estate PDFs from https://maxestates.in/downloads or copy any PDF you have:

```bash
cp /path/to/your/file.pdf data/
```

Using File Explorer: navigate to `E:\WORK\real_estate_intel\rei\data\` and paste PDFs there.

### Step 4 — Build the search index

```bash
python ingest.py
```

Expected output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Real Estate Intel — Batch Ingestion
  [1/4] Found 2 PDF(s)
  [2/4] Extracting text ...
  [3/4] Chunking ...
  [4/4] Embedding with all-mpnet-base-v2...
  ✓  Done. Index: 847 vectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 5 — Start the backend

```bash
python run.py
```

Leave this terminal running. Test it works:

```bash
curl http://localhost:5000/api/health
```

### Step 6 — Start the React frontend

Open a new terminal, navigate to the project, activate venv again:

```bash
cd /e/WORK/real_estate_intel/rei
source venv/Scripts/activate
cd frontend
npm install
npm run dev
```

Then open your browser at `http://localhost:3000`.

### Step 7 — Run the evaluation (optional)

Open a third terminal, activate venv, then:

```bash
cd /e/WORK/real_estate_intel/rei
source venv/Scripts/activate
python eval/evaluate.py
```

### Terminals at a glance

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `python run.py` | Flask API on port 5000 |
| 2 | `cd frontend && npm run dev` | React UI on port 3000 |
| 3 | `python eval/evaluate.py` | Run once for accuracy report |

### Common errors

**`(venv)` not showing after activate**
```bash
source venv/Scripts/activate
```

**`ModuleNotFoundError: No module named 'fitz'`**
```bash
pip install pymupdf
```

**`ModuleNotFoundError: No module named 'app'`**
Make sure you are in the `rei/` folder, not inside `app/`.

**`No PDFs found`**
Make sure PDF files are in `data/`, not `data/uploads/`.

**FAISS install error on Windows**
```bash
pip install faiss-cpu --no-build-isolation
```

**Port 5000 already in use**
```bash
netstat -ano | findstr :5000
taskkill /PID <number> /F
```

Docs available at `http://localhost:5000/docs`.

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/login` | None | Standard login, returns JWT |
| POST | `/api/auth/oauth/callback` | None | OAuth via Google or GitHub |
| POST | `/api/upload` | JWT | Upload and index a PDF or DOCX |
| POST | `/api/search` | JWT | Query indexed documents |

Search request:
```json
{ "query": "What is the price per sqft for Tower A?", "k": 10 }
```

Search response:
```json
{
  "answer": "Tower A is priced at [1] ...",
  "sources": [{ "ref": 1, "source": "tower-a.pdf", "page": 3, "score": 0.87 }],
  "refused": false,
  "query_type": "general",
  "latency": { "embedding": 12, "retrieval": 8, "reranking": 45, "llm": 820, "total": 887 }
}
```

When `refused` is `true`, context was insufficient and no LLM call was made.
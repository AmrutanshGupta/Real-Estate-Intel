# 🏢 Real Estate Intel: High-Precision RAG Architecture

A production-grade **Retrieval-Augmented Generation (RAG)** engine built for complex, multi-document real estate comparisons. Unlike standard vector search approaches prone to hallucinations, this engine enforces strict mathematical boundaries, multi-arm entity retrieval, and absolute logit calibration to achieve a **0.0% False Positive Rate**.

---

## ✨ Key Results

| Metric | Score |
|--------|-------|
| False Positive Rate | **0.0%** — zero hallucinations on adversarial queries |
| Entity Coverage | **84.0%** — context retrieved from multiple documents |
| Recall@3 | **78.0%** — correct answer in the top 3 chunks |
| MRR | **0.7445** — correct answer consistently ranked #1 |
| P95 Latency | **~1.5s** — fast despite heavy Cross-Encoder reranking |

---

## 📁 Project Structure

```
.
├── app/
│   ├── search_engine.py     # Orchestrates NER, multi-arm retrieval, MMR, and Cross-Encoder calibration
│   ├── vector_db.py         # FAISS (dense) + BM25 (sparse) indexes with Hard Filter per property
│   ├── pdf_loader.py        # PDF parsing and formatting artifact cleanup
│   ├── chunker.py           # Text chunking with token overlap
│   └── config.py            # Tunable parameters (MMR lambda, thresholds, chunk sizes)
│
├── eval/
│   └── evaluate.py          # ~90-query benchmark suite: Recall, MRR, nDCG, Entity Coverage, FPR
│
├── frontend/                # React/Vite UI
│
├── data/                    # Raw PDF brochures and structured JSON profiles (.gitignored)
├── faiss_index/             # Serialized FAISS indexes (.gitignored)
└── main.py                  # FastAPI entry point — /api/search and /api/upload endpoints
```

---

## 🧠 How It Works

### Core Pipeline

1. **Ingestion** — PDFs are parsed by `pdf_loader.py`, cleaned, and chunked with token overlap via `chunker.py`.
2. **Indexing** — `vector_db.py` stores chunks in both a FAISS dense index and a BM25 sparse index, with a per-property Hard Filter (`allowed_source`) to isolate document chunks.
3. **Retrieval** — `search_engine.py` runs Named Entity Recognition (NER) to detect property names, executes multi-arm retrieval for comparison queries, and applies MMR diversity filtering.
4. **Reranking** — A Cross-Encoder reranker scores candidates, with raw logits mapped to absolute probabilities via Platt Scaling (Sigmoid). Chunks scoring below the calibrated threshold (p < 0.30) are dropped.

### Multi-Arm Retrieval (for Comparisons)

When a query like *"Compare Max House and Max Towers"* is detected, the engine:
- Splits the query by entity
- Uses the Hard Filter to fetch a guaranteed quota of chunks from **each** document
- Merges results before reranking

This ensures complete coverage across documents without sacrificing precision.

---

## 🔬 Engineering Journey

Achieving zero hallucinations while maintaining high recall required three iterations of the calibration strategy:

### Option 1 — Per-Query Z-Score Normalization ❌
**Result: 100% FPR**

Z-scoring normalizes scores relative to the current candidate pool, which forces the "least bad" chunk through the threshold even on completely out-of-scope queries (e.g., *"Where are the dinosaurs?"*).

### Option 2 — Absolute Probability, Strict Threshold (p ≥ 0.5) ⚠️
**Result: 0.0% FPR, but Entity Coverage dropped to 78%**

Switching to absolute Sigmoid probabilities eliminated hallucinations entirely, but the strict threshold over-rejected valid answers when document marketing language didn't match the user's phrasing.

### Option 3 — Calibrated Threshold + Multi-Arm Retrieval ✅
**Result: 0.0% FPR + 84% Entity Coverage**

Lowering the threshold to the sweet spot of **p ≥ 0.30** and pairing it with multi-arm retrieval achieved the optimal balance — no hallucinations, high coverage, fast latency.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the frontend)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-org/real-estate-intel.git
cd real-estate-intel

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

### Running the Backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:5000`.

Endpoints:
- `POST /api/upload` — Upload and index a property PDF
- `POST /api/search` — Query across indexed properties

### Running the Frontend

```bash
cd frontend
npm run dev
```

### Running Evaluations

```bash
python eval/evaluate.py
```

Outputs Recall@3, MRR, nDCG, Entity Coverage, and False Positive Rate across ~90 benchmark queries.

---

## ⚙️ Configuration

All tunable parameters live in `app/config.py`:

| Parameter | Description |
|-----------|-------------|
| `MMR_LAMBDA` | Trade-off between relevance and diversity in MMR filtering |
| `CALIBRATION_THRESHOLD` | Minimum absolute probability for a chunk to pass reranking (default: 0.30) |
| `CHUNK_SIZE` | Token size per chunk during ingestion |
| `CHUNK_OVERLAP` | Token overlap between consecutive chunks |

---

## 🛠️ Tech Stack

- **FastAPI** — Backend API
- **FAISS** — Dense vector index
- **BM25** — Sparse keyword index
- **Cross-Encoder** — Reranking with Platt Scaling calibration
- **React + Vite** — Frontend UI

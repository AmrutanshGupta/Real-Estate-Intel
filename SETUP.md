# Setup Guide — Windows (Git Bash)

Follow these steps exactly, one command at a time.

---

## Step 1 — Create and activate virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

You should see `(venv)` appear in your terminal prompt.

---

## Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This takes 3–10 minutes the first time. It downloads PyTorch, FAISS, etc.

---

## Step 3 — Add PDF files

Copy PDF files into the `data/` folder inside the project.
You can download sample real estate PDFs from: https://maxestates.in/downloads

Or just copy any PDF you have:
```bash
cp /path/to/your/file.pdf data/
```

Using File Explorer: navigate to `E:\WORK\real_estate_intel\rei\data\` and paste PDFs there.

---

## Step 4 — Build the search index

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

---

## Step 5 — Start the Flask backend

```bash
python run.py
```

Leave this terminal running. Open a new terminal for the next step.

Test it works:
```bash
curl http://localhost:5000/api/health
```

---

## Step 6 — Start the React frontend

Open a NEW terminal window, navigate to the project, activate venv again:

```bash
cd /e/WORK/real_estate_intel/rei
source venv/Scripts/activate
cd frontend
npm install
npm run dev
```

Then open your browser: http://localhost:3000

---

## Step 7 — Run the evaluation (optional)

Open a THIRD terminal, activate venv, then:

```bash
cd /e/WORK/real_estate_intel/rei
source venv/Scripts/activate
python eval/evaluate.py
```

---

## Terminals Summary

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `python run.py` | Flask API on port 5000 |
| 2 | `cd frontend && npm run dev` | React UI on port 3000 |
| 3 | `python eval/evaluate.py` | Run once for accuracy report |

---

## Common Errors

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
Make sure PDF files are in the `data/` folder (not `data/uploads/`).

**FAISS install error on Windows**
```bash
pip install faiss-cpu --no-build-isolation
```

**Port 5000 already in use**
```bash
# Kill whatever is using it
netstat -ano | findstr :5000
# Then kill the PID shown
taskkill /PID <number> /F
```

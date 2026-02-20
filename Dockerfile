FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required dirs
RUN mkdir -p data/uploads faiss_index

ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "run.py"]

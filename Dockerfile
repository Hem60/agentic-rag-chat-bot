# ==============================================================================
# Dockerfile -- Agentic RAG Chatbot
# ==============================================================================
# One image serves the Gradio chat page and the FastAPI API on one port.
#
#   * CPU-only PyTorch (extra index in requirements.txt) keeps the image small.
#   * The embedding model and the BGE reranker are downloaded at BUILD time, so a
#     new container starts without any model download.
#   * The PDFs are copied in and src/ingest.py runs at build time, so the FAISS
#     index and BM25 chunk store are already baked in. Only build this image with
#     PDFs you are allowed to publish.
#   * Runs as a non-root user. No secret is stored in the image: pass
#     HUGGINGFACEHUB_API_TOKEN at runtime (env var / Container Apps secret).
#
# Build:  docker build -t rag-qa .
# Run:    docker run --rm -p 8000:8000 --env-file .env rag-qa
# Check:  curl http://localhost:8000/health
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=8000

# curl is only used by the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the models into the image (HF_HOME points at /app/model_cache).
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} RERANKER_MODEL=${RERANKER_MODEL}
RUN python -c "import os; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(os.environ['EMBEDDING_MODEL']); \
CrossEncoder(os.environ['RERANKER_MODEL'])"

# Application code, then the PDFs, then build the index from them.
COPY src/ ./src/
COPY app.py ./app.py
COPY data/pdfs ./data/pdfs
RUN python src/ingest.py

# Drop root. The app only needs to read /app, but HF may touch the cache dir.
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8000

# First start loads FAISS, BM25 and the reranker, so allow time.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:${PORT}/health || exit 1

# Shell form of exec so ${PORT} is expanded; `exec` keeps uvicorn as PID 1.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

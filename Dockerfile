# ==============================================================================
# Dockerfile — RAG QA System
# ==============================================================================
#
# Image size estimate:
#   Base Python slim:          ~125 MB
#   PyTorch CPU build:         ~800 MB
#   sentence-transformers:     ~500 MB
#   Other dependencies:        ~300 MB
#   Total (compressed image):  ~1.7 GB
#   Embedding model weights:   ~90 MB (downloaded at runtime, cached in volume)
#   Total on disk:             ~1.8 GB
#
# If you use Qwen/Mistral locally (fully-local mode), add ~3-4 GB for model weights.
#
# Build:   docker build -t rag-qa .
# Run:     docker-compose up
# ==============================================================================

# Use slim Python 3.11 — good balance of size and compatibility
FROM python:3.11-slim

# Metadata
LABEL maintainer="RAG QA Project"
LABEL description="RAG Question-Answering system with FAISS + Hugging Face"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# poppler-utils: needed by some PDF parsers (pdfminer)
# build-essential: needed to compile some Python packages
# We clean up apt cache after to keep the image smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
# Prevent Python from writing .pyc files (saves a bit of space)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout/stderr (so logs appear immediately)
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
# We install deps before copying code — this layer is cached by Docker
# and won't re-run on every code change, saving significant build time.
WORKDIR /app

COPY requirements.txt .

# Upgrade pip first, then install requirements
# --no-cache-dir: don't store pip's download cache in the image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy application code
# ---------------------------------------------------------------------------
COPY src/            ./src/
COPY scripts/        ./scripts/
COPY eval_data/      ./eval_data/
COPY entrypoint.sh   ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# ---------------------------------------------------------------------------
# Create directories that will be populated via volumes
# ---------------------------------------------------------------------------
RUN mkdir -p data/pdfs vectorstore model_cache

# ---------------------------------------------------------------------------
# Hugging Face cache — point to our volume-mountable directory
# This means downloaded model weights persist across container restarts
# ---------------------------------------------------------------------------
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# ---------------------------------------------------------------------------
# Expose API port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check — Docker will mark the container unhealthy if /health fails
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["./entrypoint.sh"]

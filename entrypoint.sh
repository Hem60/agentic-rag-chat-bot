#!/bin/bash
# ==============================================================================
# entrypoint.sh — Smart container startup script
# ==============================================================================
#
# This script runs when the Docker container starts.
# It:
#   1. Checks if the FAISS index exists
#   2. If not, runs ingestion automatically
#   3. Starts the FastAPI server with uvicorn
#
# Note: On the very first run, ingestion also downloads the embedding model
# (~90 MB). This is cached in the model_cache volume for future runs.
# ==============================================================================

set -e  # Exit immediately if any command fails

echo ""
echo "======================================================"
echo "  RAG QA System — Container Starting"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Check if FAISS index exists
# ---------------------------------------------------------------------------
VECTORSTORE_DIR="${VECTORSTORE_DIR:-vectorstore}"
INDEX_FILE="${VECTORSTORE_DIR}/index.faiss"

if [ -f "$INDEX_FILE" ]; then
    echo "✅  FAISS index found at: $INDEX_FILE"
    echo "    Skipping ingestion — using existing index."
else
    echo "⚠️   FAISS index not found."
    echo ""
    echo "    Checking for PDFs in data/pdfs/ ..."

    PDF_COUNT=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)

    if [ "$PDF_COUNT" -eq 0 ]; then
        echo ""
        echo "❌  ERROR: No PDFs found in data/pdfs/ AND no FAISS index exists."
        echo ""
        echo "    To fix this:"
        echo "      1. Copy your PDF files into the data/pdfs/ directory."
        echo "      2. Rebuild or restart the container."
        echo "    OR run ingestion manually:"
        echo "      docker exec -it <container_id> python src/ingest.py"
        echo ""
        echo "    Starting API anyway — /ask endpoint will return errors until"
        echo "    the index is built."
        echo ""
    else
        echo "    Found $PDF_COUNT PDF(s). Running ingestion..."
        echo "    (First run downloads embedding model — may take 2-3 minutes)"
        echo ""
        python src/ingest.py
        echo ""
        echo "✅  Ingestion complete!"
    fi
fi

echo ""
echo "🚀  Starting FastAPI server on port 8000..."
echo "    API docs: http://localhost:8000/docs"
echo "    Health:   http://localhost:8000/health"
echo ""

# ---------------------------------------------------------------------------
# Start FastAPI with uvicorn
# ---------------------------------------------------------------------------
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 matches the EXPOSE in Dockerfile and docker-compose port mapping
# --workers 1 — single worker is appropriate for CPU-bound inference
#   (multiple workers would each load a copy of the model, wasting RAM)
# --log-level info — shows request logs
exec uvicorn src.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info

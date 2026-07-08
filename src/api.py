"""
src/api.py
----------
STEP 6 — FastAPI Application

Endpoints:
  GET  /health   → returns {"status": "ok", "model": "...", "index_loaded": true/false}
  POST /ask      → accepts {"question": "..."}, returns {"answer": "...", "sources": [...]}

Run locally (no Docker) with:
    uvicorn src.api:app --reload --port 8000

Then test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "What are the main topics in the documents?"}'
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
load_dotenv()

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/gemma-2-9b-it")
VECTORSTORE_DIR  = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — loaded once at startup, reused across requests
# This avoids reloading the embedding model on every request (very slow)
# ---------------------------------------------------------------------------
app_state = {
    "vectorstore": None,
    "index_loaded": False,
    # Note: no llm object -- generation now uses InferenceClient per-request
}


# ---------------------------------------------------------------------------
# Lifespan: runs at startup and shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the vectorstore (embedding model) once at startup for fast retrieval.
    Generation uses InferenceClient called per-request -- no LLM to pre-load.
    """
    logger.info("Starting RAG QA API...")
    logger.info(f"Generation model: {GENERATION_MODEL}")

    index_file = VECTORSTORE_DIR / "index.faiss"
    if not index_file.exists():
        logger.warning(
            f"FAISS index not found at '{VECTORSTORE_DIR}/'. "
            "The /ask endpoint will return an error until you run ingestion."
        )
        app_state["index_loaded"] = False
    else:
        try:
            from src.query import load_vectorstore
            logger.info("Loading FAISS vectorstore + embedding model...")
            app_state["vectorstore"] = load_vectorstore()
            app_state["index_loaded"] = True
            logger.info("RAG pipeline ready!")
        except Exception as e:
            logger.error(f"Failed to load RAG pipeline: {e}")
            app_state["index_loaded"] = False

    yield  # Server runs here

    logger.info("Shutting down RAG QA API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG QA System",
    description=(
        "A Retrieval-Augmented Generation question-answering API. "
        "Uses local FAISS vector search + Hugging Face Inference API for generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS so you can call this from a browser-based front-end if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer using the ingested documents",
        examples=["What is the main contribution of the paper?"],
    )


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool
    vectorstore_path: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """
    Returns the API status and whether the FAISS index is loaded.
    If index_loaded is false, run ingestion before using /ask.
    """
    return HealthResponse(
        status="ok",
        model=GENERATION_MODEL,
        index_loaded=app_state["index_loaded"],
        vectorstore_path=str(VECTORSTORE_DIR),
    )


@app.post("/ask", response_model=AnswerResponse, summary="Ask a question")
async def ask(request: QuestionRequest):
    """
    Retrieve relevant document chunks and generate an answer using the RAG pipeline.

    - If the answer is in the documents, returns it with source filenames.
    - If not, the model will say it doesn't have enough information.
    """
    if not app_state["index_loaded"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS index is not loaded. Please run ingestion first:\n\n"
                "    python src/ingest.py\n\n"
                "Then restart the API server."
            ),
        )

    logger.info(f"Question received: {request.question[:80]}...")

    try:
        from src.query import answer_question
        result = answer_question(
            question=request.question,
            vectorstore=app_state["vectorstore"],
        )
        logger.info(f"Answer generated. Sources: {result['sources']}")

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            model_used=GENERATION_MODEL,
        )

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "RAG QA API is running!",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
    }

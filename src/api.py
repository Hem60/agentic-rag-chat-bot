"""
src/api.py
----------
FastAPI app: chat API plus the Gradio chat page on one port.

Endpoints:
  GET  /health   model name, index loaded, reranker loaded
  POST /chat     {question, history?} -> answer, sources (with pages), chunk previews,
                 agent trace, latency
  /ui            Gradio chat page (app.py), mounted when gradio is installed
  /              redirects to /ui

Run locally:
    uvicorn src.api:app --port 8000
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config, llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("rag.api")

CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
MAX_HISTORY_MESSAGES = 8

# Loaded once at startup, reused by every request.
app_state = {"agent": None, "retriever": None, "index_loaded": False, "reranker_loaded": False}


# ---------------------------------------------------------------------------
# Rate limiting: sliding window per client IP
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_s, clock
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG QA API (generation model: %s)", config.GENERATION_MODEL)
    try:
        from src.agent import build_agent
        from src.retriever import HybridRetriever
        retriever = HybridRetriever.from_disk()
        app_state.update(retriever=retriever, agent=build_agent(retriever),
                         index_loaded=True, reranker_loaded=retriever.reranker is not None)
        logger.info("Retriever and agent ready")
    except Exception:
        logger.exception("Failed to load the RAG pipeline")
    yield
    logger.info("Shutting down")


api = FastAPI(
    title="Agentic RAG Chatbot",
    description="Hybrid retrieval (BM25 + FAISS + RRF), BGE reranking and a LangGraph agent.",
    version="2.0.0",
    lifespan=lifespan,
)
api.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES * 4)


class SourceOut(BaseModel):
    source: str
    page: Optional[int] = None
    score: float


class ChunkOut(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    chunks: list[ChunkOut]
    trace: list[dict]
    best_score: float
    llm_calls: int
    latency_s: float


class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool
    reranker_loaded: bool


# ---------------------------------------------------------------------------
# Core logic (shared with the Gradio page)
# ---------------------------------------------------------------------------
def run_chat(question: str, history: Optional[list] = None) -> dict:
    """Run one agent turn. Raises RuntimeError if the pipeline is not loaded."""
    from src.agent import run_agent
    if app_state["agent"] is None:
        raise RuntimeError("pipeline not loaded")
    t0 = time.time()
    trimmed = (history or [])[-config.HISTORY_TURNS:]
    state = run_agent(app_state["agent"], question, trimmed)
    latency = round(time.time() - t0, 2)
    chunks = state.get("chunks") or []
    out = {
        "answer": state["answer"],
        "sources": state.get("sources", []),
        "chunks": [{"source": c["source"], "page": c["page"], "score": round(c["score"], 4),
                    "preview": c["text"][:300]} for c in chunks],
        "trace": state.get("trace", []),
        "best_score": round(state.get("best_score", 0.0), 4),
        "llm_calls": state.get("llm_calls", 0),
        "latency_s": latency,
    }
    # Log metrics only -- never the question text or answer (may be personal).
    logger.info("chat done latency=%.2fs attempts=%s llm_calls=%s best_score=%.3f",
                latency, state.get("attempts", 0), out["llm_calls"], out["best_score"])
    return out


# ---------------------------------------------------------------------------
# Endpoints (plain `def`: the work is blocking, so FastAPI runs it in a thread pool)
# ---------------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(status="ok", model=config.GENERATION_MODEL,
                          index_loaded=app_state["index_loaded"],
                          reranker_loaded=app_state["reranker_loaded"])


@api.post("/chat", response_model=ChatResponse, summary="Ask a question")
def chat(request: ChatRequest, http_request: Request):
    client = http_request.client.host if http_request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
    if not app_state["index_loaded"]:
        raise HTTPException(status_code=503,
                            detail="The knowledge base is not loaded yet. Try again shortly.")
    try:
        return run_chat(request.question, [m.model_dump() for m in request.history])
    except llm.LLMError as e:
        logger.warning("LLM unavailable: %s", e)
        raise HTTPException(status_code=503,
                            detail="The answer service is temporarily unavailable. Please try again later.")
    except Exception:
        logger.exception("Unhandled error in /chat")  # details stay in the log
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


# ---------------------------------------------------------------------------
# Gradio chat page on the same port. `app` is what uvicorn serves; `api` is the
# bare FastAPI app (used by tests). Without gradio, `app` is just `api`.
# ---------------------------------------------------------------------------
app = api
try:
    import gradio as gr
    from app import build_ui

    _password = os.getenv("APP_PASSWORD")
    app = gr.mount_gradio_app(api, build_ui(), path="/ui",
                              auth=("demo", _password) if _password else None)

    from fastapi.responses import RedirectResponse

    @api.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui")
except ImportError:  # pragma: no cover
    logger.warning("gradio not installed; serving the API only")

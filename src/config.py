# -*- coding: utf-8 -*-
"""
src/config.py
-------------
All settings read from .env in one place. Import constants from here instead of
calling os.getenv in each module.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Hugging Face / models ---
HF_TOKEN          = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
GENERATION_MODEL  = os.getenv("GENERATION_MODEL", "google/gemma-2-9b-it")
FALLBACK_MODEL    = os.getenv("FALLBACK_MODEL", "")
JUDGE_MODEL       = os.getenv("JUDGE_MODEL", "")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL    = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# --- Paths ---
PDF_DIR           = Path(os.getenv("PDF_DIR", "data/pdfs"))
VECTORSTORE_DIR   = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))

# --- Chunking / retrieval ---
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K             = int(os.getenv("TOP_K", "4"))
K_EACH            = int(os.getenv("K_EACH", "20"))
TOP_N             = int(os.getenv("TOP_N", "4"))
BM25_WEIGHT       = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT     = float(os.getenv("VECTOR_WEIGHT", "0.5"))

# --- Agent ---
RERANK_THRESHOLD  = float(os.getenv("RERANK_THRESHOLD", "0.0"))
MAX_ATTEMPTS      = int(os.getenv("MAX_ATTEMPTS", "3"))
HISTORY_TURNS     = int(os.getenv("HISTORY_TURNS", "4"))

# --- LLM call behaviour ---
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_RETRY_WAITS   = (2, 4, 8)   # seconds between attempts: exponential backoff

REFUSAL_SENTENCE = (
    "I don't have enough information in the provided documents to answer this question."
)


def token_is_set() -> bool:
    return bool(HF_TOKEN) and HF_TOKEN != "hf_YOUR_TOKEN_HERE"

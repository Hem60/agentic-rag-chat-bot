# -*- coding: utf-8 -*-
"""
src/query.py
------------
STEP 5 -- RAG Query Pipeline

This module:
  1. Loads the saved FAISS index from disk
  2. Given a question, retrieves the top-K most relevant chunks
  3. Builds a prompt instructing the model to answer ONLY from context
  4. Calls the Hugging Face Inference API for generation
  5. Returns the answer + source PDF filenames

Can be run standalone:
    python src/query.py

Or imported by the FastAPI app (src/api.py).
"""

import os
import sys
import io
from pathlib import Path
from typing import Optional

# Windows-safe UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
load_dotenv()

VECTORSTORE_DIR  = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/gemma-2-9b-it")
HF_TOKEN         = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
TOP_K            = int(os.getenv("TOP_K", "4"))


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# This prompt is carefully designed to:
# 1. Tell the model to ONLY use the provided context (no hallucination)
# 2. Give a clear fallback phrase if the answer isn't in the documents
# 3. Work well with instruction-tuned models like Mistral and Qwen
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the provided document excerpts.

Rules:
- Answer ONLY using information from the Context below.
- If the answer is not found in the Context, respond with exactly:
  "I don't have enough information in the provided documents to answer this question."
- Do NOT make up facts, dates, or numbers that aren't in the Context.
- Be concise and accurate.
- Cite the source document name when possible.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    """
    Load the FAISS index from disk.
    Raises FileNotFoundError with a helpful message if index doesn't exist.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    index_file = VECTORSTORE_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{VECTORSTORE_DIR}/'. "
            "Please run ingestion first:\n\n"
            "    python src/ingest.py\n"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # allow_dangerous_deserialization=True is required for FAISS since v0.1.17
    # It's safe here because WE created the index ourselves
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_llm():
    """
    Kept for API compatibility with api.py lifespan loader.
    Actual generation now uses call_hf_inference() directly via InferenceClient,
    which is compatible with huggingface_hub >= 0.33.
    Returns None — the llm parameter in answer_question() is no longer used.
    """
    if not HF_TOKEN or HF_TOKEN == "hf_YOUR_TOKEN_HERE":
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is not set in your .env file.\n"
            "Get a free token at: https://huggingface.co/settings/tokens\n"
            "Then set it in .env: HUGGINGFACEHUB_API_TOKEN=hf_xxxx"
        )
    # Validate token by making a quick check
    from huggingface_hub import InferenceClient
    # Just instantiate to confirm credentials — no API call yet
    _ = InferenceClient(model=GENERATION_MODEL, token=HF_TOKEN)
    return None   # actual calls use call_hf_inference()


def retrieve_chunks(vectorstore, question: str) -> list:
    """
    Retrieve the TOP_K most semantically similar chunks to the question.
    Returns a list of LangChain Document objects.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",     # cosine similarity (since we normalized embeddings)
        search_kwargs={"k": TOP_K},
    )
    docs = retriever.invoke(question)
    return docs


def format_context(docs: list) -> str:
    """
    Format retrieved chunks into a numbered context string.
    Including the source file helps the model (and you) trace which doc contributed.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        text   = doc.page_content.strip()
        parts.append(f"[Excerpt {i} -- Source: {source}, Page: {page}]\n{text}")
    return "\n\n---\n\n".join(parts)


def extract_sources(docs: list) -> list:
    """
    Extract unique source PDF filenames from the retrieved documents.
    Deduplicates so the same file isn't listed twice.
    """
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def call_hf_inference(prompt: str) -> str:
    """
    Call the Hugging Face Inference API using chat_completion.

    Why chat_completion with provider='hf-inference':
      - Modern instruct models (Mistral, Qwen, Zephyr) are chat-optimized
      - provider='hf-inference' forces HF's own free Serverless Inference API
        instead of auto-selected third-party providers (novita, together, etc.)
        which have different task routing and may not support text-generation
      - chat_completion is the stable, universal interface for instruct models

    The RAG context+question prompt is sent as the 'user' message.
    A system message reinforces the grounding rules.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN)  # auto-routes to working provider

    response = client.chat_completion(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the provided document context. "
                    "If the answer is not in the context, say: "
                    "'I don't have enough information in the provided documents to answer this question.'"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )

    # Extract the assistant's reply text
    return response.choices[0].message.content


def answer_question(question: str, vectorstore=None, llm=None) -> dict:
    """
    Main RAG pipeline: given a question, return answer + sources.

    Args:
        question:    The user's question string
        vectorstore: Pre-loaded FAISS vectorstore (loaded once, reused)
        llm:         Unused -- kept for API compatibility, actual call uses InferenceClient

    Returns:
        {
            "answer": "...",
            "sources": ["file1.pdf", "file2.pdf"],
            "retrieved_chunks": [{"source": ..., "page": ..., "text": ...}]
        }
    """
    # Load vectorstore if not already provided
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 1: Retrieve relevant chunks
    retrieved_docs = retrieve_chunks(vectorstore, question)

    # Step 2: Format context from chunks
    context = format_context(retrieved_docs)

    # Step 3: Build the full prompt string
    full_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 4: Call HF Inference API directly (avoids broken LangChain wrapper)
    raw_answer = call_hf_inference(full_prompt)

    # Step 5: Clean up the answer (some models echo parts of the prompt)
    answer = raw_answer.strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    # Step 6: Extract sources
    sources = extract_sources(retrieved_docs)

    # Step 7: Return structured result
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page":   doc.metadata.get("page", "?"),
                "text":   doc.page_content[:300] + "..."  # preview only
            }
            for doc in retrieved_docs
        ],
    }


# ---------------------------------------------------------------------------
# Standalone test (Step 5)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG QA -- Standalone Query Test")
    print("=" * 60)

    # Data Structures Unit 2 test question
    test_question = "What is a linked list and what are its types?"

    print(f"\nQuestion: {test_question}\n")
    print("Loading vectorstore and LLM...")

    try:
        vs  = load_vectorstore()
        llm = get_llm()
        print("Loaded! Querying...\n")

        result = answer_question(test_question, vectorstore=vs, llm=llm)

        print(f"Answer:\n{result['answer']}\n")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            text_preview = chunk['text'][:100].encode("ascii", "replace").decode()
            print(f"  [{i}] {chunk['source']} (page {chunk['page']}): {text_preview}...")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

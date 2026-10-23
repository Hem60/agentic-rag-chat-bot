# 🤖 RAG QA Chatbot — Data Structures Study Assistant

A **Retrieval-Augmented Generation (RAG)** question-answering system designed to parse, index, and answer questions from technical PDF documents (e.g., Data Structures lecture notes). 

Embeddings and retrieval run locally; only answer generation is hosted. This system uses a **local embedding model** running on your CPU, a **FAISS vector store** for fast semantic search, and calls the **Hugging Face Serverless Inference API** using open-weights models for high-quality, factual generation on the free tier.

---

## ✨ Features
* **Free Tier Stack:** No OpenAI, Gemini, or Pinecone keys. A free Hugging Face token is required for generation (hosted); embeddings and retrieval are local.
* **Local Embeddings:** Text chunks are embedded locally on CPU using `sentence-transformers/all-MiniLM-L6-v2`.
* **Direct Vector Storage:** Persists a FAISS index to disk so ingestion only needs to run once.
* **Structured Generation:** Uses `google/gemma-2-9b-it` via Hugging Face chat completion endpoints for accurate, grounded generation.
* **Refusal Grounding:** Built-in prompt constraints force the model to answer *only* from retrieved document excerpts or refuse with: `"I don't have enough information in the provided documents to answer this question."`
* **Evaluation Harness:** Built-in LLM-as-judge scoring script to evaluate faithfulness (1-5 scale) using a robust regex JSON parser.

---

## 📁 Project Structure

```
RAG QA/
├── data/pdfs/               ← Place your study PDFs here (e.g. Unit 2 notes)
├── vectorstore/             ← Persisted FAISS index files (auto-generated)
├── src/
│   ├── __init__.py          ← Package initialization
│   ├── ingest.py            ← PDF text extraction, chunking, and FAISS indexing
│   ├── query.py             ← RAG query pipeline (retrieval + chat completion)
│   ├── api.py               ← FastAPI REST API exposing health & query endpoints
│   └── evaluate.py          ← LLM-as-Judge evaluation harness
├── eval_data/
│   ├── test_questions.json  ← 10 evaluation test questions (with Mars refusal test)
│   └── eval_report.json     ← Generated evaluation faithfulness scores & reasoning
├── scripts/
│   └── test_hf_token.py     ← Setup verification helper
├── requirements.txt         ← CPU-only optimized Python dependencies
├── .env                     ← Project configuration and HF token (git-ignored)
└── README.md                ← This file
```

---

## 🚀 Setup & Execution Guide

### Step 1: Environment Setup
Create a virtual environment and install the CPU-optimized requirements:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install requirements (torch pinned to 2.7.1 CPU, matches numpy 2.x)
pip install -r requirements.txt
```

### Step 2: Hugging Face Configuration
1. Get a free API Token (Read access) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env   (Windows)
   cp .env.example .env     (Mac/Linux)
   ```
3. Set your token and make sure `GENERATION_MODEL` and `JUDGE_MODEL` are set to Gemma:
   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   GENERATION_MODEL=google/gemma-2-9b-it
   JUDGE_MODEL=google/gemma-2-9b-it
   ```
4. Verify your credentials using the helper script:
   ```bash
   python scripts/test_hf_token.py
   ```

### Step 3: Add PDFs
Drop your study PDFs (e.g. Data Structures slides or notes) into the `data/pdfs/` directory.

### Step 4: Run Ingestion
Load, chunk, and index your PDF contents into FAISS:
```bash
python src/ingest.py
```
*Outputs `index.faiss` and `index.pkl` in the `vectorstore/` directory.*

### Step 5: Test the Pipeline
Run a standalone query to verify the retrieval and HF inference work correctly:
```bash
python -m src.query
```

### Step 6: Start the server (API + chat page)
```bash
uvicorn src.api:app --port 8000
```
* Chat page (Gradio, with sources and an agent-trace panel): [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* `POST /chat` with `{"question": "...", "history": [...]}` returns the answer, sources with pages, chunk previews, agent trace and latency. Swagger: [/docs](http://127.0.0.1:8000/docs)
* `GET /health` reports the model, whether the index and reranker are loaded.
* Optional `.env` settings: `CORS_ORIGINS` (comma-separated), `RATE_LIMIT_PER_MIN` (default 20), `APP_PASSWORD` (enables a login, user `demo`).

---

## ⚖️ Evaluation (LLM-as-Judge)

The project includes a verification script (`src/evaluate.py`) that tests 10 questions from `eval_data/test_questions.json` (including an out-of-context refusal test). It passes them through the RAG pipeline and uses `google/gemma-2-9b-it` as an independent judge to score the answers **1 to 5** for faithfulness.

Run evaluation using:
```bash
python -m src.evaluate
```

The script saves a JSON report to `eval_data/eval_report.json` with per-question scores, reasoning, and latency stats.

### Reading the report honestly

A question can fail before it ever reaches the judge (HF rate limits, timeouts). Those are recorded with `"score": null` and counted separately, so the report always reconciles:

```
questions_attempted  = questions_scored + questions_errored
score_distribution   sums to questions_scored
```

Two averages are reported because they answer different questions:

| Field | Meaning |
|---|---|
| `average_score` | Mean over **scored questions only**. Quote it as *"X/5 over N scored questions."* |
| `average_score_penalised` | Same run with every pipeline error counted as a 1 — the end-to-end view. |
| `judge_parse_fallbacks` | Scores recovered by regex because the judge returned malformed JSON. Less trustworthy. |

The committed report is a real run: **10 attempted, 6 scored, 4 errored** — `average_score` 3.5 over those 6, `average_score_penalised` 2.5 over all 10. Earlier versions of this harness reported only the 3.5 next to a count of 10, which read as a result for the whole set.

---

## 🐳 Docker Deployment

The project is fully prepared for containerized deployment. 

### Why CPU-only is enforced:
The `requirements.txt` is configured with `--extra-index-url https://download.pytorch.org/whl/cpu` to ensure the container downloads only the **lightweight CPU-only PyTorch build** (~180MB) instead of the standard GPU/CUDA builds (~2.5GB). This prevents out-of-memory (OOM) crashes on local Docker Desktop setups.

### Running with Docker Compose:
1. Rebuild the container:
   ```bash
   docker build -t rag-qa .
   ```
2. Start the service:
   ```bash
   docker-compose up
   ```
The server will start on port `8000`, mounting `data/pdfs/` and `vectorstore/` as persistent volumes.

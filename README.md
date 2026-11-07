# Agentic RAG Chatbot — Data Structures & Algorithms Assistant

A chatbot that answers **Data Structures and Algorithms** questions from a knowledge base made of your own lecture-note PDFs plus 72 Wikipedia articles on the subject (fetched automatically, so nothing has to be uploaded). Answers always come from retrieved text; questions on other topics are politely declined. It combines **hybrid retrieval** (BM25 keyword search + FAISS vector search, merged with Reciprocal Rank Fusion), a **BGE cross-encoder reranker**, and a **LangGraph agent** that rewrites questions, checks result quality, retries with a new query, and refuses when the knowledge base does not contain the answer. Every answer lists its sources (PDF file and page, or the Wikipedia article).

**What runs where.** Embeddings, retrieval and reranking run locally on CPU. Only answer generation is hosted: it calls the Hugging Face Inference API (or any OpenAI-compatible provider, see Configuration) with a model you choose in `.env`. There is no OpenAI or Pinecone dependency.

> **Status.** Code, API, chat UI, tests, the full 3-variant evaluation and a Dockerfile are in place. The Dockerfile has not been built successfully on the development machine yet, and no cloud deployment has been done.

---

## How it works (in simple words)

**Step 0, done once: build the knowledge base.** The bot reads the lecture PDFs and 72 Wikipedia articles on Data Structures, cuts them into small pieces, and saves them so they can be searched quickly.

```mermaid
flowchart LR
    A[PDFs and Wikipedia articles] --> B[Cut into small pieces]
    B --> C[(Searchable knowledge base)]
```

**Every time someone asks a question:**

```mermaid
flowchart TD
    Q([You ask a question]) --> T{Is it just hello or thanks?}
    T -- Yes --> H[Friendly reply]
    T -- No --> F[Find the best matching pieces in the knowledge base]
    F --> OK{Are the pieces relevant enough?}
    OK -- Yes --> A[The AI writes the answer using ONLY those pieces]
    A --> S([You get the answer and its sources])
    OK -- Not really --> R[Reword the question and search again, up to 3 tries]
    R --> OK
    OK -- Nothing related at all --> N([Polite refusal: I only answer Data Structures questions])
```

What each box means:

| Step | What happens | Why it matters |
|---|---|---|
| Find the best matching pieces | Two searches run together: one by exact words, one by meaning. Their results are merged, then a stronger model re-checks the top ones and keeps the best 4. | Finds the right text even if you phrase the question differently. |
| Relevant enough? | The bot scores how well the pieces match your question. | Stops it from answering when it has nothing useful. |
| Reword and search again | A follow-up like "what are its disadvantages?" is rewritten using the chat history into a full question. Weak searches are retried with different words. | Makes follow-up questions and unusual phrasing work. |
| Write the answer | The AI is told to use only the retrieved pieces and to say so if they do not contain the answer. | Keeps answers grounded in real sources instead of guesses. |
| Sources | Every answer lists where it came from (PDF page or Wikipedia article). | You can check the answer yourself. |

Example: *"How does a hash table handle collisions?"* finds the Hash table article, the AI summarises it (separate chaining, open addressing), and the answer shows `Wikipedia: Hash table` as the source. *"Who won the 2011 Cricket World Cup?"* matches nothing, so the bot politely declines.

## Architecture (technical detail)

```mermaid
flowchart TD
    Q[User question + last chat messages] --> R{route}
    R -- greeting / thanks --> S[canned reply, no LLM]
    R -- document question --> W[rewrite: standalone query<br/>skipped on first try with no history]
    W --> RET[retrieve]
    subgraph RET_BOX [Hybrid retriever, runs locally]
      B[BM25 top 20] --> F[RRF fusion]
      V[FAISS vector top 20] --> F
      F --> X[BGE cross-encoder rerank, keep top 4]
    end
    RET --> B
    RET --> V
    X --> G{grade: best rerank score >= threshold?}
    G -- yes --> GEN[generate grounded answer with sources + pages]
    G -- no, attempts < 3 --> W
    G -- no, attempts exhausted --> REF[refuse with the fixed sentence]
```

Grading uses the reranker score, not another LLM call, which keeps free-tier usage low: 1 LLM call for a clear question, 2 for a follow-up, about 4 in the worst retry case.

Offline (run once per set of PDFs): PDF text extraction (pypdf, pdfminer fallback) → text cleaning → chunking → local embeddings → FAISS index + `chunks.json` for BM25, saved in `vectorstore/`.

## Project layout

```
app.py                  Gradio chat page (mounted into FastAPI)
src/
  config.py             all settings read from .env
  fetch_corpus.py       downloads the Wikipedia DS/Algorithms articles into data/corpus/
  ingest.py             load PDFs + corpus, clean, chunk, embed, save index + chunks.json
  retriever.py          BM25 + vector + RRF + BGE reranker
  llm.py                HF chat call with retry, backoff, fallback model
  agent.py              LangGraph agent (route, rewrite, retrieve, grade, generate, refuse)
  api.py                FastAPI: /chat, /health, plus the chat page
  calibrate.py          picks RERANK_THRESHOLD from data
  evaluate.py           3-variant evaluation + LLM judge
  metrics.py            hit@k, MRR, refusal and summary helpers
  query.py              the original naive pipeline (evaluation baseline A)
tests/                  pytest suite (no network or models needed)
eval_data/              42 test questions with gold sources, calibration set, reports
data/pdfs/              your lecture-note PDFs
data/corpus/            Wikipedia articles as .txt (CC BY-SA, see data/corpus/README.md)
Dockerfile, .dockerignore, docker-compose.yml
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt  # CPU-only PyTorch via the extra index in the file
copy .env.example .env           # Windows   (Mac/Linux: cp .env.example .env)
```

Edit `.env`: set `HUGGINGFACEHUB_API_TOKEN` (free "Read" token from huggingface.co/settings/tokens), `GENERATION_MODEL`, an optional `FALLBACK_MODEL`, and a `JUDGE_MODEL` that is **different** from the generator. Then check the token and build the index:

```bash
python scripts/test_hf_token.py
python -m src.fetch_corpus       # optional: re-download the Wikipedia articles (already in data/corpus/)
python src/ingest.py             # writes vectorstore/index.faiss, index.pkl, chunks.json
```

Add your own PDFs to `data/pdfs/` and re-run `ingest.py` to include them; the corpus articles are indexed alongside.

## Run

```bash
uvicorn src.api:app --port 8000
```

| URL | What |
|---|---|
| `http://127.0.0.1:8000/` (redirects to `/ui`) | Chat page: sources under each answer, collapsible agent trace |
| `POST /chat` | `{"question": "...", "history": [{"role": "user", "content": "..."}]}` → answer, sources with pages, chunk previews, trace, latency |
| `GET /health` | model name, index loaded, reranker loaded |
| `/docs` | Swagger UI |

The first start loads FAISS, BM25 and the reranker (tens of seconds on CPU). Each retrieval reranks up to 40 candidates on CPU, about 6–10 s per query on the development laptop.

Safety knobs in `.env`: `CORS_ORIGINS` (comma-separated, default localhost only), `RATE_LIMIT_PER_MIN` (default 20 per IP on `/chat`), `APP_PASSWORD` (enables a login for the chat page, user `demo`). Errors return generic messages; details go to the server log, and question text is never logged.

## Configuration

All variables are optional except the token. See `.env.example` for comments.

| Variable | Default | Purpose |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | – | HF Inference API access |
| `GENERATION_MODEL` / `FALLBACK_MODEL` | gemma-2-9b-it / – | answer model; fallback used when it is not served |
| `LLM_BASE_URL` / `LLM_API_KEY` | – | optional: use any OpenAI-compatible provider (Groq, Gemini, OpenRouter...) instead of Hugging Face, e.g. when HF credits run out (HTTP 402). The model variables then name that provider's models |
| `JUDGE_MODEL` | – | evaluation judge, must differ from the generator |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | local embeddings |
| `RERANKER_MODEL` | BAAI/bge-reranker-base | cross-encoder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | chunking |
| `K_EACH` / `TOP_N` | 20 / 4 | candidates per retriever / final chunks |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | 0.5 / 0.5 | fusion weights |
| `RERANK_THRESHOLD` | 0.0 (`.env.example`: 0.3) | below this the agent retries, then refuses; a score below a tenth of it refuses at once |
| `MAX_ATTEMPTS` / `HISTORY_TURNS` | 3 / 4 | searches per question / messages the agent sees |

### Calibrating the threshold

`RERANK_THRESHOLD` should come from your data, not a guess:

```bash
python -m src.calibrate
```

It scores 32 answerable and 15 unanswerable questions (`eval_data/calibration_questions.json`) and suggests a value. On the current knowledge base it suggested 0.64 (answerable scores were all at least 0.675), but two off-topic questions scored high (a TCP question 0.998, a boiling-point question 0.607), so no threshold separates everything. I set 0.3 instead: a lower value avoids wrongly refusing real Data Structures questions, and the answer prompt also refuses off-topic questions as a second guard. That is a judgement call, not a measured optimum.

## Evaluation

`eval_data/test_questions.json` has 42 questions: 33 answerable and 9 off-topic. 21 answerable questions come from the lecture notes (19 standalone, 2 follow-ups with chat history; gold = the right PDF page or the matching Wikipedia article) and 12 are general Data Structures questions answered only by the Wikipedia corpus (gold = that article). Three variants run on the same questions:

| Variant | What it is |
|---|---|
| A: Naive | vector search only, top 4, straight to the LLM (the original system) |
| B: Hybrid + rerank | BM25 + vector + RRF + BGE reranker, top 4, straight to the LLM |
| C: Agentic | the full LangGraph agent |

```bash
python -m src.evaluate                    # all variants + judge (needs HF credits)
python -m src.evaluate --retrieval-only   # hit@4 and MRR for A and B, no LLM calls
python -m src.evaluate --variants A,B --no-judge
```

Metrics: hit@4 and MRR (is a gold page among the final 4 chunks), faithfulness (1–5 from an LLM judge), refusal accuracy, wrong refusals, errors, latency and LLM calls. Each average is shown with the number of questions behind it; a pipeline error is counted as an error, never dropped or scored as a 1. Judge scores are directional only — spot-check about 10 answers by hand.

### Results

Real run of `python -m src.evaluate` on the current system: knowledge base = lecture notes + 72 Wikipedia articles, 42 questions (33 answerable, 9 off-topic), `RERANK_THRESHOLD=0.3`. Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b` (a different model), both via Groq. **0 pipeline errors in every variant.**

| Variant | Hit@4 | MRR | Faithfulness (1-5) | Refusal correct | Wrong refusals | Errors | Avg latency | Avg LLM calls |
|---|---|---|---|---|---|---|---|---|
| A: Naive | 100% (33/33) | 0.95 | 4.63 (n=30) | 100% (9/9) | 3/33 | 0/42 | 7.4 s | 1.00 |
| B: Hybrid + rerank | 100% (33/33) | 0.91 | 4.87 (n=31) | 100% (9/9) | 2/33 | 0/42 | 13.6 s | 1.00 |
| C: Agentic | 100% (33/33) | 0.91 | 4.97 (n=32) | 100% (9/9) | 1/33 | 0/42 | 11.3 s | 0.98 |

Faithfulness is judged only for answers a variant actually gave, so n is smaller than 33 where it refused an answerable question. "Wrong refusals" counts answerable questions the bot declined.

**What the numbers say**

- **The naive baseline is strong here.** With the Wikipedia articles in the index, plain vector search already finds the right source for every question (hit@4 100%) and even has the best MRR (0.95). The reranker did *not* improve retrieval on this set. B and C ranked the labelled article second on four corpus questions (quicksort, Dijkstra, BFS/DFS, binary search), putting another closely related article first. Part of this is how I labelled gold (one specific article per question), so treat the MRR gap as weak evidence, not a defect proven.
- **Where B and C do better is answer quality.** Faithfulness goes 4.63 (A) → 4.87 (B) → 4.97 (C), and wrong refusals go 3 → 2 → 1 of 33. Differences of this size on 33 questions are small and could be noise.
- **A refused three answerable questions**, including two plain stack questions (stack operations, the underflow check) and one follow-up; the old naive prompt is more eager to say "not enough information". B refused the "it" follow-up (no query rewriting) and, like C, the memoization question: both returned four chunks from the Memoization article only, none of which explained memoization in dynamic programming, so the generator declined. A's mix of four different articles answered it. C rewrites follow-ups, so it answered both follow-up questions.
- **Refusal on off-topic questions is 9/9 for every variant**, so the score gate did not change refusal accuracy on this set; the prompt rule alone was enough for this generator. Off-topic questions such as TCP/UDP or a boiling point can still score high in the reranker, which is why the prompt refusal remains a second guard.
- **Latency:** every variant is dominated by the hosted LLM call (about 7 s for A, whose retrieval takes well under a second). Reranking on CPU adds roughly 5-6 s (B vs A). C averaged less than B here because the near-zero-score early exit refuses off-topic questions without extra searches.

**Limits, stated plainly**

- 33 answerable and 9 off-topic questions is small, and I wrote them myself from the same sources. Most are easy and well covered, so most differences between variants are within noise.
- Judge scores come from one small model and are directional. I read the refused answerable questions and the lowest scores by hand; the other answers were not individually read.
- Latencies are noisy (shared machine, remote LLM). Treat them as ordering, not benchmarks.
- Retrieval is scored by gold source (pypdf 0-indexed pages for the PDFs, a whole article for Wikipedia); the app shows 1-indexed PDF pages.

#### Earlier run (notes-only index, 30 questions)

Kept for context: before the Wikipedia corpus, the scope prompt and the 0.3 threshold existed, on the lecture notes alone. It does **not** describe the current system.

| Variant | Hit@4 | MRR | Faithfulness | Wrong refusals |
|---|---|---|---|---|
| A: Naive | 95% (20/21) | 0.81 | 4.81 (n=21) | 0/21 |
| B: Hybrid + rerank | 100% (21/21) | 0.93 | 5.00 (n=19) | 2/21 |
| C: Agentic | 100% (21/21) | 0.95 | 5.00 (n=21) | 0/21 |

On the small notes-only index the reranker clearly helped retrieval (MRR 0.81 → 0.93); on the larger, easier-to-search corpus that advantage disappeared. The agent's clearest benefit in both runs is follow-up questions.

## Tests

```bash
python -m pytest -q
```

67 tests, no network or model downloads: text cleaning, retrieval and fusion logic, agent routing (greeting, retry, refusal after 3 weak searches, fallback query), the LLM wrapper's retry/backoff/fallback with a mocked client, API behaviour (sources with pages, safe errors, rate limit, CORS), judge parsing and metrics.

## Docker

```bash
docker build -t rag-qa .
docker run --rm -p 8000:8000 --env-file .env rag-qa
curl http://localhost:8000/health
```

The image installs CPU-only PyTorch, downloads the embedding model and the BGE reranker at **build** time, copies `data/pdfs/` and `data/corpus/` and runs `src/ingest.py` during the build, so the index is baked in and containers start without downloads. It runs as a non-root user, keeps `.env` out of the image (`.dockerignore`), and takes the token only at runtime. `PORT` (default 8000) sets the listening port. Expect an image of roughly 2 GB. Only build with PDFs you are allowed to publish.

`docker compose up --build` does the same with `.env` loaded; it deliberately mounts no volumes, since a mounted `vectorstore/` would hide the baked-in index.

> The Dockerfile has not been built successfully on the development machine yet: three attempts failed while downloading Python packages (PyPI read timeouts), before any of the project's own build steps ran. Treat it as untested until `docker build` and a `/health` check pass.

## Deploy on Render

[render.yaml](render.yaml) describes one Docker web service that serves the chat page and the API.

1. Push the repo to GitHub, then in Render choose **New → Blueprint** and select it.
2. Enter the two secrets it asks for: `LLM_API_KEY` (your OpenAI-compatible provider key) and `APP_PASSWORD` (the chat page login, user `demo`; keep it set on a public demo so strangers cannot spend your provider quota).
3. After the first deploy, set `CORS_ORIGINS` to your `https://<name>.onrender.com` URL and redeploy.
4. Check `/health` (`index_loaded` and `reranker_loaded` should be true), then open `/`.

The service needs about 2 GB of RAM (PyTorch + embedding model + reranker), so Render's free 512 MB tier will not work; the blueprint uses the `standard` plan, so confirm current plans and prices first. The first build downloads PyTorch and the models and can take 10-20 minutes. The Dockerfile has not been built successfully on the development machine yet, so this build is its first real test: read the build log if it fails.

## Planned next

Cloud deployment (Render or Hugging Face Spaces; the earlier Azure plan was dropped), faster reranking (fewer candidates or a smaller reranker), a larger and harder test set, and the ablations (chunk size, k, BM25 weight, embedding model).

## Limits

- Answers depend on a small hosted model; faithfulness scores come from another small model and are directional.
- Retrieval quality was measured on 21 questions from one document set.
- The reranker is slow on CPU and overconfident on some out-of-scope questions (see calibration).
- Scanned PDFs without a text layer are not supported (no OCR).

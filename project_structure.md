# RAG QA Project Structure

rag-qa/
├── data/
│   └── pdfs/                    # Drop your 3-5 PDFs here
├── vectorstore/                 # FAISS index saved here after ingestion
├── src/
│   ├── __init__.py
│   ├── ingest.py                # Step 4: PDF ingestion + FAISS index builder
│   ├── query.py                 # Step 5: RAG query pipeline
│   ├── api.py                   # Step 6: FastAPI app
│   └── evaluate.py              # Step 8: Evaluation harness
├── eval_data/
│   └── test_questions.json      # Step 7: 10 test questions
├── scripts/
│   └── test_hf_token.py         # Step 2: Token verification script
├── .env.example                 # Step 2: Config template
├── .env                         # Step 2: Your actual secrets (git-ignored)
├── requirements.txt             # Step 1: All dependencies
├── Dockerfile                   # Step 9: Container definition
├── docker-compose.yml           # Step 9: Compose config
├── entrypoint.sh                # Step 9: Smart startup script
└── .gitignore                   # Excludes .env, vectorstore, model cache

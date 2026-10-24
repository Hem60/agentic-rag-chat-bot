# -*- coding: utf-8 -*-
"""
src/ingest.py
-------------
STEP 4 -- PDF Ingestion Pipeline

This script:
  1. Scans data/pdfs/ for all PDF files
  2. Loads and parses each PDF using PyPDF (with pdfminer fallback)
  3. Splits text into overlapping chunks
  4. Embeds chunks locally using sentence-transformers (no API key needed)
  5. Saves a FAISS vector index to disk

Run with:
    python src/ingest.py

First run is slower -- it downloads the embedding model (~90 MB for all-MiniLM-L6-v2).
Subsequent runs use the cached model and are much faster.
"""

import os
import sys
import json
import re
import time
import unicodedata
from pathlib import Path

# Windows-safe UTF-8 output (prevents emoji encoding errors on cp1252 terminals)
# reconfigure() edits the streams in place and is skipped when they don't support it
# (e.g. under pytest capture); replacing sys.stdout at import time broke test runs.
for _stream in (sys.stdout, sys.stderr):
    if sys.platform == "win32" and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load configuration from .env
# ---------------------------------------------------------------------------
load_dotenv()

PDF_DIR         = Path(os.getenv("PDF_DIR", "data/pdfs"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "150"))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
# PDFs that use Symbol/Wingdings for bullets encode them in the Unicode Private
# Use Area (U+E000-U+F8FF). pypdf passes those codepoints through verbatim, so
# chunks end up holding glyphs like U+F0B7 that mean nothing to the embedding
# model and leak into the context we hand the LLM. Map the common ones to their
# obvious ASCII equivalent and drop whatever is left.
PUA_REPLACEMENTS = {
    "": "- ",   # Symbol font bullet
    "": "- ",   # Wingdings square bullet
    "": "- ",   # Wingdings check mark
    "": "- ",   # Wingdings arrowhead bullet
    "": "-",    # Symbol font hyphen
    "": "->",   # Wingdings right arrow
}

_PUA_RE        = re.compile(r"[-]")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_TRAIL_WS_RE   = re.compile(r"[ \t]+(?=\n)")
_SPACES_RE     = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """
    Normalise raw PDF text before chunking.

    1. Replace known Private Use Area bullet glyphs with ASCII equivalents.
    2. Strip any remaining PUA codepoints (unmapped Symbol/Wingdings glyphs).
    3. NFKC-normalise, which folds PDF ligatures (fi, fl) back into plain
       letters so retrieval matches the words a user actually types.
    4. Collapse runs of whitespace introduced by the PDF's column layout.
    """
    for glyph, replacement in PUA_REPLACEMENTS.items():
        text = text.replace(glyph, replacement)

    text = _PUA_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)

    text = _TRAIL_WS_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)

    return text.strip()


def count_pua(text: str) -> int:
    """Number of Private Use Area codepoints in a string (for reporting)."""
    return len(_PUA_RE.findall(text))


def load_pdfs(pdf_dir: Path) -> list:
    """
    Load all PDFs from the given directory.
    Uses PyPDFLoader (backed by pypdf) which handles most modern PDFs well.
    Falls back to PDFMinerLoader for PDFs that pypdf can't parse.

    Returns a list of LangChain Document objects.
    """
    from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[ERROR] No PDF files found in '{pdf_dir}/'")
        print("    Drop 3-5 PDF files into that folder, then re-run ingestion.\n")
        sys.exit(1)

    print(f"\n[INFO] Found {len(pdf_files)} PDF(s) in '{pdf_dir}/':")
    for p in pdf_files:
        size_kb = p.stat().st_size / 1024
        print(f"      - {p.name}  ({size_kb:.1f} KB)")

    all_docs = []
    pua_found = 0
    for pdf_path in pdf_files:
        print(f"\n    Loading: {pdf_path.name} ...", end="", flush=True)
        try:
            # PyPDFLoader preserves page numbers in metadata
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            print(f" OK  ({len(docs)} page(s))")
        except Exception as e:
            print(f" [WARN] PyPDF failed ({e}), trying PDFMiner ...")
            try:
                loader = PDFMinerLoader(str(pdf_path))
                docs = loader.load()
                print(f"     PDFMiner succeeded: {len(docs)} page(s)")
            except Exception as e2:
                print(f"     [ERROR] PDFMiner also failed: {e2} -- skipping this file.")
                continue

        # Make sure every document carries the source filename, and normalise
        # the extracted text before it ever reaches the splitter.
        for doc in docs:
            doc.metadata.setdefault("source", pdf_path.name)
            pua_found += count_pua(doc.page_content)
            doc.page_content = clean_text(doc.page_content)

        all_docs.extend(docs)

    print(f"\n    Total pages loaded: {len(all_docs)}")
    if pua_found:
        print(f"    Normalised {pua_found} Private Use Area glyph(s) (Symbol/Wingdings bullets)")
    return all_docs


def split_documents(docs: list) -> list:
    """
    Split documents into chunks.

    chunk_size=1000 and chunk_overlap=150 work well for most technical/research PDFs.
    Larger chunks = more context per retrieval but slower embedding.
    Smaller chunks = faster but may lose cross-sentence context.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print(f"\n[INFO] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split priority: paragraph -> newline -> sentence -> word -> character
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    print(f"    Total chunks created: {len(chunks)}")

    # Show a sample chunk so you can sanity-check the split quality
    if chunks:
        sample = chunks[0]
        print(f"\n    Sample chunk (chunk 0):")
        print(f"    Source: {sample.metadata.get('source', 'unknown')}")
        print(f"    Length: {len(sample.page_content)} chars")
        # stdout is already wrapped as UTF-8 above, so print the real text --
        # the old ascii/replace coercion was hiding bad glyphs behind '?'.
        preview = sample.page_content[:200].strip()
        print(f"    Preview: {preview}...")

    return chunks


def build_vectorstore(chunks: list) -> None:
    """
    Embed all chunks and build a FAISS index.

    The embedding model runs 100% locally -- no API key, no internet required
    after the first download.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print(f"\n[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    print("    (First run downloads ~90 MB; subsequent runs use cache)")

    t0 = time.time()

    # model_kwargs={'device': 'cpu'} forces CPU even if CUDA is available
    # encode_kwargs={'normalize_embeddings': True} improves cosine similarity quality
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"    Embedding model loaded in {time.time() - t0:.1f}s")
    print(f"\n[INFO] Embedding {len(chunks)} chunks (this may take a minute) ...")

    t1 = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - t1
    print(f"    Embedding complete in {elapsed:.1f}s ({len(chunks)/elapsed:.0f} chunks/sec)")

    # Save the index to disk
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    # Verify the files were written
    saved_files = list(VECTORSTORE_DIR.iterdir())
    print(f"\n[SAVED] FAISS index saved to '{VECTORSTORE_DIR}/'")
    for f in saved_files:
        size_kb = f.stat().st_size / 1024
        print(f"      - {f.name}  ({size_kb:.1f} KB)")


def save_chunks(chunks: list) -> None:
    """Save chunk text + metadata so BM25 can be rebuilt at startup without re-reading PDFs."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    path = VECTORSTORE_DIR / "chunks.json"
    data = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {len(data)} chunks for BM25 -> {path}")


def main():
    print("=" * 60)
    print("  RAG QA -- PDF Ingestion Pipeline")
    print("=" * 60)

    # Sanity check: PDF directory must exist
    if not PDF_DIR.exists():
        print(f"\n[ERROR] Directory '{PDF_DIR}' does not exist.")
        print("    Run this from the project root, not from inside src/.\n")
        sys.exit(1)

    start = time.time()

    docs   = load_pdfs(PDF_DIR)
    chunks = split_documents(docs)
    build_vectorstore(chunks)

    print(f"\n[DONE] Ingestion complete in {time.time() - start:.1f}s")
    print("    You can now run the query pipeline or start the API.\n")


if __name__ == "__main__":
    main()

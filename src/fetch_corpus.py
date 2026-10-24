# -*- coding: utf-8 -*-
"""
src/fetch_corpus.py
-------------------
Build a broad Data Structures & Algorithms knowledge base so the bot can answer
DS questions that are not covered by your own PDFs, without uploading anything.

Downloads the plain text of a fixed list of Wikipedia articles into data/corpus/
(one .txt per article). Run once, then re-run ingestion:

    python -m src.fetch_corpus
    python src/ingest.py

Wikipedia text is licensed CC BY-SA 4.0. Each saved file keeps its title and URL,
and the chatbot cites it as "Wikipedia: <title>". Attribution notes are written
to data/corpus/README.md.
"""

import re
import sys
import time
from pathlib import Path

import httpx

CORPUS_DIR = Path("data/corpus")
API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves.
USER_AGENT = "RAG-QA-StudyBot/1.0 (https://github.com/Hem60/RAG-CHAT-BOT; educational RAG project)"

TOPICS = [
    # basics
    "Data structure", "Abstract data type", "Algorithm", "Array (data structure)", "Dynamic array",
    "Linked list", "Doubly linked list", "Circular buffer", "Stack (abstract data type)",
    "Queue (abstract data type)", "Double-ended queue", "Priority queue", "Sparse matrix",
    # hashing and sets
    "Hash table", "Hash function", "Hash collision", "Bloom filter", "Disjoint-set data structure",
    # trees
    "Tree (abstract data type)", "Binary tree", "Binary search tree", "AVL tree", "Red–black tree",
    "B-tree", "B+ tree", "Trie", "Heap (data structure)", "Binary heap", "Segment tree",
    "Fenwick tree", "Splay tree", "Treap", "Skip list", "Tree traversal",
    # graphs
    "Graph (abstract data type)", "Adjacency list", "Adjacency matrix", "Breadth-first search",
    "Depth-first search", "Dijkstra's algorithm", "Bellman–Ford algorithm",
    "Floyd–Warshall algorithm", "Prim's algorithm", "Kruskal's algorithm", "Topological sorting",
    "Minimum spanning tree",
    # sorting and searching
    "Sorting algorithm", "Bubble sort", "Selection sort", "Insertion sort", "Merge sort",
    "Quicksort", "Heapsort", "Counting sort", "Radix sort", "Shellsort", "Binary search",
    "Linear search",
    # analysis and techniques
    "Big O notation", "Time complexity", "Space complexity", "Amortized analysis",
    "Recursion (computer science)", "Dynamic programming", "Memoization", "Greedy algorithm",
    "Divide-and-conquer algorithm", "Backtracking", "Tower of Hanoi",
    # expressions
    "Reverse Polish notation", "Shunting yard algorithm", "Infix notation",
]

# Sections that add nothing to question answering.
_SKIP_SECTIONS = {"references", "external links", "further reading", "see also", "notes",
                  "bibliography", "sources", "footnotes", "citations"}
_HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower().replace("+", "plus")).strip("_")


def strip_sections(text: str) -> str:
    """Drop References / External links / See also ... sections from plain text."""
    out, skipping, last = [], False, 0
    for m in _HEADING_RE.finditer(text):
        if not skipping:
            out.append(text[last:m.start()])
        skipping = m.group(2).strip().lower() in _SKIP_SECTIONS
        last = m.end()
    if not skipping:
        out.append(text[last:])
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def fetch_article(client: httpx.Client, title: str) -> dict:
    r = client.get(API, params={
        "action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1,
        "redirects": 1, "format": "json", "formatversion": 2, "titles": title,
    })
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        raise LookupError(f"no article for {title!r}")
    return {"title": page["title"], "url": page["fullurl"], "text": strip_sections(page["extract"])}


def save_article(art: dict, directory: Path = CORPUS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify(art['title'])}.txt"
    # Header lines are read back by ingest.py (title + url), then a blank line, then the body.
    path.write_text(f"TITLE: {art['title']}\nURL: {art['url']}\n\n{art['text']}\n", encoding="utf-8")
    return path


def write_readme(directory: Path = CORPUS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text(
        "# Corpus attribution\n\n"
        "The `.txt` files here are plain-text extracts of English Wikipedia articles, fetched by\n"
        "`python -m src.fetch_corpus`. Text is available under the Creative Commons\n"
        "Attribution-ShareAlike 4.0 license (https://creativecommons.org/licenses/by-sa/4.0/).\n"
        "Each file starts with the article title and URL; authors are listed in the article\n"
        "history on Wikipedia. Changes made: references / external-links sections removed and\n"
        "markup flattened to plain text.\n", encoding="utf-8")


def main() -> int:
    ok, failed = 0, []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        for i, title in enumerate(TOPICS, 1):
            try:
                art = fetch_article(client, title)
                path = save_article(art)
                print(f"[{i:>2}/{len(TOPICS)}] {art['title']:<38} {len(art['text']):>7} chars -> {path.name}")
                ok += 1
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                print(f"[{i:>2}/{len(TOPICS)}] FAILED {title!r}: {type(e).__name__}: {e}")
                failed.append(title)
            time.sleep(0.3)  # be polite to the API
    write_readme()
    print(f"\nSaved {ok} article(s) to {CORPUS_DIR}/" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())

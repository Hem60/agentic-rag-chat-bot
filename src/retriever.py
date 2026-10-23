# -*- coding: utf-8 -*-
"""
src/retriever.py
----------------
Hybrid retrieval: BM25 + FAISS vector search merged with Reciprocal Rank Fusion,
then re-scored by a cross-encoder reranker.

Built once at startup, called many times:

    retriever = HybridRetriever.from_disk()
    hits = retriever.search("what is stack underflow?")
    # -> [{"text", "source", "page", "score", "rank"}, ...]  best first

The vector search, chunk list and reranker are injectable so the logic can be
unit-tested without downloading any model.
"""

import json
import logging
import re
from typing import Callable, Optional, Sequence

from src import config

log = logging.getLogger("rag.retriever")

RRF_K = 60  # standard RRF damping constant

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens; keeps terms like 'avl' or 'lifo' intact."""
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: Sequence[Sequence[int]], weights: Sequence[float], k: int = RRF_K) -> list:
    """
    Weighted Reciprocal Rank Fusion.

    `rankings` holds one best-first list of chunk ids per retriever. A chunk's
    fused score is sum(weight / (k + rank)), so one ranked high in several lists
    rises to the top. Returns chunk ids, best first.
    """
    scores: dict = {}
    for ranking, weight in zip(rankings, weights):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        chunks: list,
        vector_search: Callable[[str, int], list],
        reranker=None,
        k_each: int = config.K_EACH,
        top_n: int = config.TOP_N,
        bm25_weight: float = config.BM25_WEIGHT,
        vector_weight: float = config.VECTOR_WEIGHT,
    ):
        """
        chunks:        list of {"text", "metadata"} dicts; index in the list is the chunk id.
        vector_search: fn(query, k) -> list of chunk ids, best first.
        reranker:      object with predict([(query, text), ...]) -> scores; None skips reranking.
        """
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.vector_search = vector_search
        self.reranker = reranker
        self.k_each = k_each
        self.top_n = top_n
        self.weights = (bm25_weight, vector_weight)
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    # -- candidate generation ------------------------------------------------
    def bm25_search(self, query: str, k: int) -> list:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [i for i in order if scores[i] > 0]

    def candidates(self, query: str) -> list:
        """Chunk ids from BM25 + vector search merged with RRF."""
        bm25_ids = self.bm25_search(query, self.k_each)
        vec_ids = self.vector_search(query, self.k_each)
        return rrf_fuse([bm25_ids, vec_ids], self.weights)

    def _hit(self, i: int, score: float, rank: int) -> dict:
        meta = self.chunks[i]["metadata"]
        return {"text": self.chunks[i]["text"], "source": meta.get("source", "unknown"),
                "page": meta.get("page"), "score": score, "rank": rank}

    def vector_only(self, query: str, k: Optional[int] = None) -> list:
        """Plain vector search, no BM25 / rerank. Used as the naive baseline in evaluation."""
        ids = self.vector_search(query, k or self.top_n)
        return [self._hit(i, 0.0, r) for r, i in enumerate(ids, start=1)]

    # -- full search ---------------------------------------------------------
    def search(self, query: str) -> list:
        ids = self.candidates(query)
        if not ids:
            return []

        if self.reranker is not None:
            scores = [float(s) for s in self.reranker.predict(
                [(query, self.chunks[i]["text"]) for i in ids])]
            ranked = sorted(zip(scores, ids), key=lambda x: x[0], reverse=True)
        else:
            # No reranker: keep fusion order; score is not comparable across queries.
            ranked = [(0.0, i) for i in ids]

        return [self._hit(i, score, rank)
                for rank, (score, i) in enumerate(ranked[: self.top_n], start=1)]

    # -- construction from the saved index -------------------------------------
    @classmethod
    def from_disk(cls, reranker_name: Optional[str] = None, use_reranker: bool = True, **kw):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        store_dir = config.VECTORSTORE_DIR
        if not (store_dir / "index.faiss").exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{store_dir}/'. Run: python src/ingest.py")

        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        store = FAISS.load_local(str(store_dir), embeddings, allow_dangerous_deserialization=True)

        chunks_file = store_dir / "chunks.json"
        if chunks_file.exists():
            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        else:
            log.warning("chunks.json missing; rebuilding chunk list from the FAISS docstore")
            chunks = [{"text": d.page_content, "metadata": d.metadata}
                      for d in store.docstore._dict.values()]

        # Map FAISS result documents back to chunk ids by (source, page, text).
        key_to_id = {(c["metadata"].get("source"), c["metadata"].get("page"), c["text"]): i
                     for i, c in enumerate(chunks)}

        def vector_search(query: str, k: int) -> list:
            ids = []
            for d in store.similarity_search(query, k=k):
                cid = key_to_id.get((d.metadata.get("source"), d.metadata.get("page"), d.page_content))
                if cid is not None:
                    ids.append(cid)
            return ids

        reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(reranker_name or config.RERANKER_MODEL, device="cpu")

        return cls(chunks, vector_search, reranker, **kw)

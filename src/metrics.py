# -*- coding: utf-8 -*-
"""
src/metrics.py
--------------
Pure metric helpers for the evaluation harness (no models, easy to unit test).
Page numbers follow the pypdf convention used in chunk metadata (0-indexed).
"""

from typing import Optional

from src import config


def _norm(source: str) -> str:
    return (source or "").replace("\\", "/").split("/")[-1]


def gold_rank(chunks: list, gold: list) -> Optional[int]:
    """1-based rank of the first retrieved chunk on a gold page, or None.

    chunks: best-first list of {"source", "page", ...}
    gold:   [{"source": "x.pdf", "pages": [3, 4]}, ...]
    """
    targets = {(_norm(g["source"]), p) for g in gold for p in (g.get("pages") or [])}
    whole_sources = {_norm(g["source"]) for g in gold if not g.get("pages")}  # "pages": null = any page
    for rank, c in enumerate(chunks, start=1):
        src = _norm(c.get("source"))
        if src in whole_sources or (src, c.get("page")) in targets:
            return rank
    return None


def hit_at_k(chunks: list, gold: list, k: int = 4) -> bool:
    r = gold_rank(chunks[:k], gold)
    return r is not None


def reciprocal_rank(chunks: list, gold: list, k: int = 4) -> float:
    r = gold_rank(chunks[:k], gold)
    return 1.0 / r if r else 0.0


def is_refusal(answer: str) -> bool:
    return config.REFUSAL_SENTENCE.lower() in (answer or "").lower()


def _mean(xs: list) -> Optional[float]:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list) -> dict:
    """
    Aggregate per-question rows for one variant.

    Each row: {"answerable": bool, "error": bool, "hit": bool|None, "rr": float|None,
               "refused": bool|None, "faithfulness": int|None, "latency_s": float,
               "llm_calls": int}
    Every average carries the count it was computed over, so a number is never
    quoted for more questions than actually contributed to it.
    """
    ans = [r for r in rows if r["answerable"]]
    non = [r for r in rows if not r["answerable"]]
    ret = [r for r in ans if r.get("hit") is not None]
    gen_ans = [r for r in ans if r.get("refused") is not None]
    gen_non = [r for r in non if r.get("refused") is not None]
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    done = [r for r in rows if not r["error"]]
    return {
        "questions": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "hit_at_4": _mean([1.0 if r["hit"] else 0.0 for r in ret]),
        "hit_at_4_n": len(ret),
        "mrr": _mean([r["rr"] for r in ret]),
        "faithfulness": _mean(faith),
        "faithfulness_n": len(faith),
        "refusal_correct": (sum(r["refused"] for r in gen_non) / len(gen_non)) if gen_non else None,
        "refusal_correct_n": len(gen_non),
        "wrong_refusals": sum(r["refused"] for r in gen_ans),
        "wrong_refusals_n": len(gen_ans),
        "avg_latency_s": _mean([r["latency_s"] for r in done]),
        "avg_llm_calls": _mean([r["llm_calls"] for r in done]),
    }

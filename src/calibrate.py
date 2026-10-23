# -*- coding: utf-8 -*-
"""
src/calibrate.py
----------------
Pick RERANK_THRESHOLD from data instead of guessing.

Runs every question in eval_data/calibration_questions.json through the hybrid
retriever, records the best reranker score, and chooses the threshold that
best separates answerable questions (should score above it) from unanswerable
ones (should score below it).

    python -m src.calibrate

Writes eval_data/calibration.json and prints the suggested value for .env.
"""

import json
import sys
from pathlib import Path

QUESTIONS_PATH = Path("eval_data/calibration_questions.json")
REPORT_PATH = Path("eval_data/calibration.json")


def best_threshold(answerable: list, unanswerable: list) -> dict:
    """
    Choose t so that score >= t means "answer". Maximises balanced accuracy
    (mean of the answerable pass rate and the unanswerable refusal rate); ties
    go to the t in the widest gap between the two groups.
    """
    if not answerable or not unanswerable:
        raise ValueError("need at least one answerable and one unanswerable score")
    pts = sorted(set(answerable) | set(unanswerable))
    cands = [pts[0] - 1e-6] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1e-6]

    best = None
    for t in cands:
        tpr = sum(s >= t for s in answerable) / len(answerable)
        tnr = sum(s < t for s in unanswerable) / len(unanswerable)
        bal = (tpr + tnr) / 2
        lo = max([s for s in pts if s < t], default=t)
        hi = min([s for s in pts if s >= t], default=t)
        key = (bal, hi - lo)
        if best is None or key > best[0]:
            best = (key, t, tpr, tnr)
    (bal, _), t, tpr, tnr = best
    return {"threshold": round(t, 4), "balanced_accuracy": round(bal, 3),
            "answerable_pass_rate": round(tpr, 3), "unanswerable_refusal_rate": round(tnr, 3)}


def main() -> None:
    from src.retriever import HybridRetriever

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"[INFO] {len(questions)} calibration questions; loading retriever ...")
    retriever = HybridRetriever.from_disk()

    rows = []
    for q in questions:
        hits = retriever.search(q["question"])
        top = hits[0] if hits else None
        rows.append({
            "question": q["question"],
            "answerable": q["answerable"],
            "best_score": top["score"] if top else 0.0,
            "top_source": top["source"] if top else None,
            "top_page": top["page"] if top else None,
            "top_preview": top["text"][:120].replace("\n", " ") if top else "",
        })
        tag = "ANS " if q["answerable"] else "NOAN"
        print(f"  {tag} {rows[-1]['best_score']:.3f}  p.{rows[-1]['top_page']}  {q['question'][:70]}")

    ans = [r["best_score"] for r in rows if r["answerable"]]
    non = [r["best_score"] for r in rows if not r["answerable"]]
    result = best_threshold(ans, non)
    report = {**result, "n_answerable": len(ans), "n_unanswerable": len(non),
              "answerable_min": min(ans), "unanswerable_max": max(non), "rows": rows}
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAnswerable scores   : min {min(ans):.3f}  max {max(ans):.3f}")
    print(f"Unanswerable scores : min {min(non):.3f}  max {max(non):.3f}")
    print(f"Balanced accuracy   : {result['balanced_accuracy']}  "
          f"(answered {result['answerable_pass_rate']:.0%} of answerable, "
          f"refused {result['unanswerable_refusal_rate']:.0%} of unanswerable)")
    print(f"\nSuggested: RERANK_THRESHOLD={result['threshold']}   (add to .env)")
    print(f"[SAVED] {REPORT_PATH} -- spot-check top_preview for answerable rows scoring low.")


if __name__ == "__main__":
    sys.exit(main())

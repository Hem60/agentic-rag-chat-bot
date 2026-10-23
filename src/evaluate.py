"""
src/evaluate.py
---------------
Three-variant evaluation with an LLM-as-judge.

Variants, all run on the same questions:
  A  naive      vector search only, top 4, straight to the LLM (the original system)
  B  hybrid     BM25 + vector + RRF + BGE rerank, top 4, straight to the LLM
  C  agentic    full LangGraph agent: rewrite, grade, retry, refuse

Metrics per variant (see src/metrics.py): hit@4, MRR, faithfulness (judge,
1-5), refusal accuracy, wrong refusals, errors, latency, LLM calls.

Run:
    python -m src.evaluate                      # everything (needs HF credits)
    python -m src.evaluate --retrieval-only     # A and B retrieval metrics, no LLM
    python -m src.evaluate --variants A,B       # subset
    python -m src.evaluate --no-judge           # skip faithfulness scoring

Outputs eval_data/eval_report.json and eval_data/eval_table.md.

Reading the numbers honestly
  * Every average is printed with the number of questions behind it.
  * A question that errors in the pipeline is counted as an error, not dropped
    and not scored as a 1.
  * Faithfulness is judged only for answers the system actually gave (refusals
    are measured by refusal accuracy). Judge scores are directional: small free
    models are imperfect judges, so spot-check about 10 answers by hand.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src import config, llm
from src.metrics import hit_at_k, is_refusal, reciprocal_rank, summarize

JUDGE_MODEL = config.JUDGE_MODEL
HF_TOKEN = config.HF_TOKEN
EVAL_DATA_PATH = Path("eval_data/test_questions.json")
REPORT_PATH = Path("eval_data/eval_report.json")
TABLE_PATH = Path("eval_data/eval_table.md")

VARIANT_NAMES = {"A": "A: Naive (vector top 4)", "B": "B: Hybrid + rerank", "C": "C: Agentic"}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a RAG (Retrieval-Augmented Generation) system.

Your task: Score the Generated Answer on a scale of 1-5 for FAITHFULNESS.

Faithfulness means: Is the answer grounded in and supported by the Retrieved Context?
- The answer should NOT contain information beyond what is in the Context.
- If the question cannot be answered from the Context, the system should say so.

Scoring Rubric:
5 = Perfect: Answer is completely accurate, directly supported by Context, no hallucinations.
4 = Good: Mostly accurate, minor gaps or vague phrasing, but no fabrications.
3 = Partial: Some correct points, but also some inaccuracies or unjustified claims.
2 = Poor: Mostly incorrect or contains fabricated details not in Context.
1 = Fail: Completely wrong, hallucinated, or refused to answer when Context had the answer.

Special case for refusal:
- If the Context does NOT contain the answer AND the Generated Answer correctly says it doesn't have enough information -> Score 5.
- If the Context does NOT contain the answer BUT the Generated Answer hallucinated an answer -> Score 1.

---

Question: {question}

Expected Answer (ground truth): {expected_answer}

Retrieved Context:
{context}

Generated Answer: {generated_answer}

---

Respond ONLY in this exact JSON format (no extra text):
{{
  "score": <integer 1-5>,
  "reasoning": "<one or two sentences explaining the score>"
}}"""


def call_judge_inference(prompt: str) -> str:
    """Call the judge model through src.llm (retry + backoff). No fallback model:
    silently switching judges mid-run would make scores incomparable."""
    return llm.chat(
        [
            {"role": "system", "content": "You are an expert RAG evaluator. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model=JUDGE_MODEL,
        max_tokens=256,
        temperature=0.01,
        fallback_model="",
    )


def judge_answer(question: str, expected_answer: str, generated_answer: str, context: str) -> dict:
    """
    Use the judge LLM to score the generated answer.
    Returns {"score": int | None, "reasoning": str, "parse_failed": bool}

    `parse_failed` marks answers whose score came from the regex fallback rather
    than clean JSON. `score` is None when the judge reply contained no usable
    score at all; such a question is reported as unscored instead of getting a 1.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, expected_answer=expected_answer,
        context=context, generated_answer=generated_answer)

    raw = call_judge_inference(prompt).strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    if "{" in raw and "}" in raw:
        raw = raw[raw.index("{"): raw.rindex("}") + 1]

    parse_failed = False
    try:
        result = json.loads(raw)
        score = max(1, min(5, int(result["score"])))
        reasoning = result.get("reasoning", "No reasoning provided.")
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        import re
        parse_failed = True
        score = None
        reasoning = "No reasoning provided."

        score_match = re.search(r'"?score"?\s*:\s*(\d)', raw, re.IGNORECASE)
        if score_match:
            score = max(1, min(5, int(score_match.group(1))))

        reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).replace('\\"', '"')
        else:
            reasoning_match = re.search(r'"?reasoning"?\s*:\s*(.*)', raw, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip('",{}[] ')
        if not score_match:
            reasoning = f"Could not parse judge response: {raw[:200]}"

    return {"score": score, "reasoning": reasoning, "parse_failed": parse_failed}


def format_context_for_eval(chunks: list) -> str:
    """Full chunk text for the judge (never truncated)."""
    if not chunks:
        return "No context retrieved."
    return "\n\n".join(f"[{c.get('source', '?')} p.{c.get('page', '?')}]: {c.get('text', '')}"
                       for c in chunks)


# ---------------------------------------------------------------------------
# Variants. Each returns {"chunks": [...], "answer": str|None, "llm_calls": int}
# ---------------------------------------------------------------------------
def make_variants(retriever, which: list, generate: bool) -> dict:
    from src import agent as agent_mod
    from src.query import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT, clean_answer

    def gen_a(question, history):
        chunks = retriever.vector_only(question, config.TOP_K)
        answer = None
        if generate:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=agent_mod.format_context(chunks), question=question)
            answer = clean_answer(llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                            {"role": "user", "content": prompt}]))
        return {"chunks": chunks, "answer": answer, "llm_calls": 1 if generate else 0}

    def gen_b(question, history):
        chunks = retriever.search(question)
        answer = None
        if generate:
            prompt = agent_mod.ANSWER_PROMPT.format(
                refusal=config.REFUSAL_SENTENCE,
                context=agent_mod.format_context(chunks), question=question)
            answer = agent_mod.clean_answer(llm.chat([{"role": "user", "content": prompt}]))
        return {"chunks": chunks, "answer": answer, "llm_calls": 1 if generate else 0}

    agent_graph = agent_mod.build_agent(retriever) if "C" in which else None

    def gen_c(question, history):
        state = agent_mod.run_agent(agent_graph, question, history)
        return {"chunks": state.get("chunks") or [], "answer": state["answer"],
                "llm_calls": state.get("llm_calls", 0)}

    return {"A": gen_a, "B": gen_b, "C": gen_c}


def load_questions(limit=None) -> list:
    if not EVAL_DATA_PATH.exists():
        sys.exit(f"[ERROR] Test questions file not found: '{EVAL_DATA_PATH}'")
    questions = json.loads(EVAL_DATA_PATH.read_text(encoding="utf-8"))
    return questions[:limit] if limit else questions


def eval_question(q: dict, fn, generate: bool, judge: bool) -> dict:
    """Run one question through one variant. Never raises: failures become error rows."""
    row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
           "error": False, "error_detail": None, "hit": None, "rr": None, "refused": None,
           "faithfulness": None, "judge_reasoning": None, "judge_parse_failed": False,
           "answer": None, "sources": [], "latency_s": 0.0, "llm_calls": 0}
    t0 = time.time()
    try:
        out = fn(q["question"], q.get("history") or [])
        row["llm_calls"] = out["llm_calls"]
        chunks = out["chunks"]
        row["sources"] = [{"source": c["source"], "page": c["page"]} for c in chunks]
        if q["answerable"]:
            row["hit"] = hit_at_k(chunks, q["gold"])
            row["rr"] = reciprocal_rank(chunks, q["gold"])
        if generate:
            row["answer"] = out["answer"]
            row["refused"] = is_refusal(out["answer"])
            if judge and not row["refused"]:
                j = judge_answer(q["question"], q["expected_answer"], out["answer"],
                                 format_context_for_eval(chunks))
                row["faithfulness"] = j["score"]
                row["judge_reasoning"] = j["reasoning"]
                row["judge_parse_failed"] = j["parse_failed"]
    except Exception as e:  # noqa: BLE001 - recorded per question, run continues
        row["error"] = True
        row["error_detail"] = f"{type(e).__name__}: {e}"
    row["latency_s"] = round(time.time() - t0, 2)
    return row


def fmt(v, pct=False, n=None) -> str:
    if v is None:
        return "n/a"
    s = f"{v:.0%}" if pct else f"{v:.2f}"
    return f"{s} (n={n})" if n is not None else s


def build_table(summaries: dict) -> str:
    head = ("| Variant | Hit@4 | MRR | Faithfulness (1-5) | Refusal correct | Wrong refusals "
            "| Errors | Avg latency (s) | Avg LLM calls |\n|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for v, s in summaries.items():
        rows.append(
            f"| {VARIANT_NAMES[v]} | {fmt(s['hit_at_4'], True, s['hit_at_4_n'])} | {fmt(s['mrr'])} "
            f"| {fmt(s['faithfulness'], n=s['faithfulness_n'])} "
            f"| {fmt(s['refusal_correct'], True, s['refusal_correct_n'])} "
            f"| {s['wrong_refusals']}/{s['wrong_refusals_n']} | {s['errors']}/{s['questions']} "
            f"| {fmt(s['avg_latency_s'])} | {fmt(s['avg_llm_calls'])} |")
    return head + "\n".join(rows) + "\n"


def run_evaluation(variants: list, retrieval_only: bool, judge: bool, limit=None) -> dict:
    generate = not retrieval_only
    if retrieval_only:
        variants = [v for v in variants if v != "C"]  # the agent needs the LLM to rewrite/answer
    judge = judge and generate

    if judge:
        if not JUDGE_MODEL:
            sys.exit("[ERROR] JUDGE_MODEL is not set in .env.")
        if JUDGE_MODEL == config.GENERATION_MODEL:
            sys.exit("[ERROR] JUDGE_MODEL equals GENERATION_MODEL. A model grading its own answers "
                     "inflates faithfulness scores; set a different JUDGE_MODEL.")

    questions = load_questions(limit)
    print("=" * 66)
    print(f"  Evaluation: {len(questions)} questions, variants {','.join(variants)}"
          + ("  [retrieval only]" if retrieval_only else ""))
    print("=" * 66)

    from src.retriever import HybridRetriever
    print("[INFO] Loading retriever ...")
    retriever = HybridRetriever.from_disk()
    fns = make_variants(retriever, variants, generate)

    all_rows, summaries = {}, {}
    for v in variants:
        print(f"\n--- Variant {VARIANT_NAMES[v]} ---")
        rows = []
        for q in questions:
            row = eval_question(q, fns[v], generate, judge)
            rows.append(row)
            tag = "ERR " if row["error"] else ("hit " if row["hit"] else ("miss" if row["hit"] is False else "    "))
            extra = f" faith={row['faithfulness']}" if row["faithfulness"] else ""
            extra += " refused" if row["refused"] else ""
            print(f"  Q{q['id']:>2} {tag} {row['latency_s']:>6.1f}s{extra}  {q['question'][:55]}")
            if row["error"]:
                print(f"       {row['error_detail'][:110]}")
        all_rows[v] = rows
        summaries[v] = summarize(rows)

    table = build_table(summaries)
    print("\n" + table)

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "retrieval_only" if retrieval_only else "full",
        "generation_model": config.GENERATION_MODEL,
        "judge_model": JUDGE_MODEL if judge else None,
        "rerank_threshold": config.RERANK_THRESHOLD,
        "n_questions": len(questions),
        "n_answerable": sum(q["answerable"] for q in questions),
        "n_unanswerable": sum(not q["answerable"] for q in questions),
        "summary": summaries,
        "results": all_rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    TABLE_PATH.write_text(table, encoding="utf-8")
    print(f"[SAVED] {REPORT_PATH} and {TABLE_PATH}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--retrieval-only", action="store_true", help="skip all LLM calls")
    ap.add_argument("--no-judge", action="store_true", help="skip faithfulness scoring")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    a = ap.parse_args(argv)
    variants = [v.strip().upper() for v in a.variants.split(",") if v.strip()]
    if not set(variants) <= set(VARIANT_NAMES):
        sys.exit("--variants must be a subset of A,B,C")
    run_evaluation(variants, a.retrieval_only, not a.no_judge, a.limit)


if __name__ == "__main__":
    main()

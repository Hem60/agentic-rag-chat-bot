"""
src/evaluate.py
---------------
STEP 8 — Evaluation Harness with LLM-as-Judge

This script:
  1. Loads test questions from eval_data/test_questions.json
  2. Runs each question through the RAG pipeline
  3. Uses an LLM (same HF model, or a separate judge) to score faithfulness 1-5
  4. Prints a per-question report + overall average

Run with:
    python src/evaluate.py

Output: prints to terminal AND saves a JSON report to eval_data/eval_report.json

Scoring rubric (faithfulness 1-5):
  5 — Answer is fully correct, clearly derived from context
  4 — Answer is mostly correct with minor gaps
  3 — Partially correct or somewhat vague
  2 — Mostly incorrect or fabricated details
  1 — Completely wrong or the model hallucinated when it should have refused

For the refusal test (Q10), scoring is:
  5 — Model correctly said it doesn't have enough information
  1 — Model hallucinated an answer instead of refusing

Note on score quality:
  Smaller free HF models (Mistral 7B, Qwen 7B) are a realistic but imperfect judge.
  They score ~70-80% correlation with GPT-4-class judges. This is the key tradeoff
  of using free models — GPT-4o-mini as judge would give more reliable scores.
  Use these scores for directional feedback, not absolute truth.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

from src import config, llm

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
JUDGE_MODEL      = config.JUDGE_MODEL
HF_TOKEN         = config.HF_TOKEN
EVAL_DATA_PATH   = Path("eval_data/test_questions.json")
REPORT_PATH      = Path("eval_data/eval_report.json")


# ---------------------------------------------------------------------------
# Judge prompt template
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
- If the Context does NOT contain the answer AND the Generated Answer correctly says it doesn't have enough information → Score 5.
- If the Context does NOT contain the answer BUT the Generated Answer hallucinated an answer → Score 1.

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


def load_test_questions() -> list:
    """Load and validate test questions from the JSON file."""
    if not EVAL_DATA_PATH.exists():
        print(f"\n[ERROR] Test questions file not found: '{EVAL_DATA_PATH}'")
        sys.exit(1)

    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Check for unfilled placeholders
    unfilled = [q for q in questions if "PLACEHOLDER" in q.get("question", "")]
    if unfilled:
        print(f"\n⚠️   WARNING: {len(unfilled)} question(s) still have PLACEHOLDER text.")
        print("    Edit eval_data/test_questions.json with real questions from your PDFs.")
        print("    Continuing with non-placeholder questions only...\n")
        questions = [q for q in questions if "PLACEHOLDER" not in q.get("question", "")]

    if not questions:
        print("\n❌  No valid questions found (all were placeholders). Edit the file first.\n")
        sys.exit(1)

    print(f"[INFO] Loaded {len(questions)} test question(s)")
    return questions


def call_judge_inference(prompt: str) -> str:
    """Call the judge model through src.llm (retry + backoff). No fallback model:
    silently switching judges mid-run would make scores incomparable."""
    return llm.chat(
        [
            {"role": "system", "content": "You are an expert RAG evaluator. Respond only with valid JSON."},
            {"role": "user",   "content": prompt},
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

    `score` is None when the judge reply contained no usable score at all. Such
    a question is reported as unscored instead of being silently given a 1.

    `parse_failed` marks answers whose score came from the regex fallback rather
    than clean JSON. Those scores are less trustworthy, so the summary counts
    them separately instead of blending them silently into the average.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        context=context,
        generated_answer=generated_answer,
    )

    raw = call_judge_inference(prompt).strip()
    # Some models wrap JSON in markdown code blocks
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    # Find the JSON object if the model added extra text
    if "{" in raw and "}" in raw:
        raw = raw[raw.index("{"): raw.rindex("}") + 1]

    parse_failed = False
    try:
        result = json.loads(raw)
        score = max(1, min(5, int(result["score"])))   # Clamp to 1-5
        reasoning = result.get("reasoning", "No reasoning provided.")
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        # Fallback if the model doesn't return valid JSON (common for 7B/9B instruct models)
        import re
        parse_failed = True
        score = None
        reasoning = "No reasoning provided."

        # Search for score key and value (flexible with quotes)
        score_match = re.search(r'"?score"?\s*:\s*(\d)', raw, re.IGNORECASE)
        if score_match:
            score = max(1, min(5, int(score_match.group(1))))

        # Search for reasoning key and value (flexible with quotes)
        reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).replace('\\"', '"')
        else:
            # Secondary check if quote is missing at the start/end
            reasoning_match = re.search(r'"?reasoning"?\s*:\s*(.*)', raw, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip('",{}[] ')

        # If we failed to parse even a score via regex, fall back to showing raw output
        if not score_match:
            reasoning = f"Could not parse judge response: {raw[:200]}"

    return {"score": score, "reasoning": reasoning, "parse_failed": parse_failed}



def format_context_for_eval(chunks: list) -> str:
    """Format retrieved chunks for the judge prompt."""
    if not chunks:
        return "No context retrieved."
    parts = []
    for c in chunks:
        parts.append(f"[{c.get('source', '?')} p.{c.get('page', '?')}]: {c.get('text', '')}")
    return "\n\n".join(parts)


def run_evaluation():
    """Run the full evaluation harness."""
    print("=" * 65)
    print("  RAG QA — Evaluation Harness")
    print(f"  Judge model: {JUDGE_MODEL}")
    print("=" * 65)

    # Load questions
    questions = load_test_questions()

    # Load RAG pipeline
    print("\n[INFO] Loading RAG pipeline...")
    try:
        from src.query import load_vectorstore, answer_question, format_context
        vectorstore = load_vectorstore()
        # Validate HF token early
        if not HF_TOKEN or HF_TOKEN == "hf_YOUR_TOKEN_HERE":
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in .env")
        print("    Pipeline ready.\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Run each question.
    # `scores` holds only questions the judge actually scored; `results` holds
    # every question including pipeline errors. Keep the two apart -- averaging
    # over `scores` while reporting len(results) is what made an earlier report
    # claim 10 questions at 3.5/5 when only 6 had been scored.
    results = []
    scores  = []
    errors  = []

    for i, q in enumerate(questions, 1):
        qid      = q.get("id", i)
        question = q["question"]
        expected = q["expected_answer"]

        print(f"[{i}/{len(questions)}] Q{qid}: {question[:60]}...")

        try:
            # Get RAG answer
            t0 = time.time()
            rag_result = answer_question(question, vectorstore=vectorstore)
            elapsed    = time.time() - t0

            generated = rag_result["answer"]
            sources   = rag_result["sources"]
            context   = format_context_for_eval(rag_result["retrieved_chunks"])

            # Judge the answer
            judgment = judge_answer(question, expected, generated, context)
            score    = judgment["score"]
            if score is None:
                raise ValueError(f"Judge returned no usable score: {judgment['reasoning']}")
            scores.append(score)
            if judgment["parse_failed"]:
                print("    [WARN] Judge response was not valid JSON; score recovered by regex.")

            status = "[OK]" if score >= 4 else ("[PARTIAL]" if score == 3 else "[FAIL]")
            print(f"    {status}  Score: {score}/5 | Time: {elapsed:.1f}s | Sources: {sources}")
            print(f"       Reasoning: {judgment['reasoning'][:100]}...")

            results.append({
                "id":               qid,
                "question":         question,
                "expected_answer":  expected,
                "generated_answer": generated,
                "sources":          sources,
                "score":            score,
                "judge_parse_failed": judgment["parse_failed"],
                "reasoning":        judgment["reasoning"],
                "latency_s":        round(elapsed, 2),
            })

        except Exception as e:
            print(f"    [ERROR] {e}")
            errors.append(qid)
            results.append({
                "id":               qid,
                "question":         question,
                "expected_answer":  expected,
                "generated_answer": f"ERROR: {str(e)}",
                "sources":          [],
                # null, not 0 -- this question was never scored, and a 0 here
                # reads as "the judge gave it zero", which is a different claim.
                "score":            None,
                "judge_parse_failed": False,
                "reasoning":        f"Pipeline error: {str(e)}",
                "latency_s":        0,
            })

        print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    avg_score  = sum(scores) / len(scores) if scores else 0
    score_dist = {i: scores.count(i) for i in range(1, 6)}
    parse_fails = sum(1 for r in results if r.get("judge_parse_failed"))

    # Two averages, because they answer different questions. `avg_score` is
    # "how good were the answers we managed to score". `avg_penalised` treats
    # every pipeline error as a 1 -- a question the system failed to answer at
    # all is a failure of the system, not a question that doesn't count.
    penalised     = scores + [1] * len(errors)
    avg_penalised = sum(penalised) / len(penalised) if penalised else 0

    print("=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Questions attempted : {len(results)}")
    print(f"  Questions scored    : {len(scores)}")
    print(f"  Pipeline errors     : {len(errors)}" + (f"  (Q{', Q'.join(str(e) for e in errors)})" if errors else ""))
    print(f"  Judge parse fallback: {parse_fails}")
    print(f"  Average (scored)    : {avg_score:.2f} / 5.00  over {len(scores)} question(s)")
    print(f"  Average (errors=1)  : {avg_penalised:.2f} / 5.00  over {len(results)} question(s)")
    print(f"  Score distribution  : {score_dist}  (sums to {sum(score_dist.values())})")
    print(f"  Score breakdown     :")
    print(f"    5/5 (perfect)     : {score_dist.get(5, 0)}")
    print(f"    4/5 (good)        : {score_dist.get(4, 0)}")
    print(f"    3/5 (partial)     : {score_dist.get(3, 0)}")
    print(f"    2/5 (poor)        : {score_dist.get(2, 0)}")
    print(f"    1/5 (fail)        : {score_dist.get(1, 0)}")
    print(f"    unscored (error)  : {len(errors)}")

    if errors:
        print()
        print(f"  [WARN] {len(errors)} of {len(results)} question(s) never reached the judge.")
        print("         Quote the average as 'X/5 over N scored questions', not as a")
        print("         result for the full set.")

    # Interpret scores
    print()
    if avg_score >= 4.0:
        print("  [EXCELLENT] Your RAG pipeline is performing very well!")
    elif avg_score >= 3.0:
        print("  [GOOD] Solid performance, some room for improvement.")
    elif avg_score >= 2.0:
        print("  [FAIR] Consider tuning chunk size, overlap, or top-K.")
    else:
        print("  [POOR] Check ingestion quality, prompt, and model choice.")

    # Debugging tips
    print()
    print("  [DEBUG] Debugging tips if scores are low:")
    print("     - Bad chunking? Reduce CHUNK_SIZE or increase CHUNK_OVERLAP in .env")
    print("     - Weak retrieval? Increase TOP_K (e.g., 6 or 8) in .env")
    print("     - Model struggling? This is a known tradeoff with free 7B models vs")
    print("       GPT-4o-mini. 7B models follow instructions less reliably.")
    print("       Try: HuggingFaceH4/zephyr-7b-beta or Qwen/Qwen2.5-7B-Instruct")
    print("     - PDF quality? Check if your PDFs have selectable text (not scanned images)")

    # Save report
    report = {
        "timestamp":    datetime.now().isoformat(),
        "judge_model":  JUDGE_MODEL,
        "questions_attempted": len(results),
        "questions_scored":    len(scores),
        "questions_errored":   len(errors),
        "errored_ids":         errors,
        "judge_parse_fallbacks": parse_fails,
        # Average over scored questions only. Read alongside questions_scored --
        # it is not a result for the full question set unless the two match.
        "average_score":            round(avg_score, 3),
        "average_score_basis":      "scored questions only",
        # Same run, with each pipeline error counted as a 1.
        "average_score_penalised":  round(avg_penalised, 3),
        "score_distribution": score_dist,
        "results":      results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Full report saved to: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    run_evaluation()

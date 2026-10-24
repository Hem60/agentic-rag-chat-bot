from src import config
from src.metrics import gold_rank, hit_at_k, is_refusal, reciprocal_rank, summarize

GOLD = [{"source": "a.pdf", "pages": [3, 4]}]


def ch(source, page):
    return {"source": source, "page": page}


def test_gold_rank_matches_source_and_page_ignoring_dirs():
    chunks = [ch(r"data\pdfs\b.pdf", 3), ch(r"data\pdfs\a.pdf", 9), ch("data/pdfs/a.pdf", 4)]
    assert gold_rank(chunks, GOLD) == 3          # b.pdf p.3 is the wrong file


def test_hit_and_rr_respect_k():
    chunks = [ch("a.pdf", 1), ch("a.pdf", 2), ch("a.pdf", 5), ch("a.pdf", 6), ch("a.pdf", 3)]
    assert not hit_at_k(chunks, GOLD, 4) and hit_at_k(chunks, GOLD, 5)
    assert reciprocal_rank(chunks, GOLD, 4) == 0.0
    assert reciprocal_rank([ch("a.pdf", 4)], GOLD) == 1.0
    assert reciprocal_rank([ch("a.pdf", 1), ch("a.pdf", 3)], GOLD) == 0.5


def test_is_refusal():
    assert is_refusal(config.REFUSAL_SENTENCE)
    assert is_refusal("Sorry. " + config.REFUSAL_SENTENCE.upper())
    assert not is_refusal("A stack is LIFO.") and not is_refusal(None)


def _row(**kw):
    base = {"answerable": True, "error": False, "hit": None, "rr": None, "refused": None,
            "faithfulness": None, "latency_s": 1.0, "llm_calls": 1}
    return {**base, **kw}


def test_summarize_counts_denominators_and_errors():
    rows = [
        _row(hit=True, rr=1.0, refused=False, faithfulness=4),
        _row(hit=False, rr=0.0, refused=True),                      # wrong refusal
        _row(error=True, latency_s=0.0, llm_calls=0),               # errored, no retrieval
        _row(answerable=False, refused=True),
        _row(answerable=False, refused=False, faithfulness=1),
    ]
    s = summarize(rows)
    assert s["questions"] == 5 and s["errors"] == 1
    assert s["hit_at_4"] == 0.5 and s["hit_at_4_n"] == 2 and s["mrr"] == 0.5
    assert s["faithfulness"] == 2.5 and s["faithfulness_n"] == 2
    assert s["refusal_correct"] == 0.5 and s["refusal_correct_n"] == 2
    assert s["wrong_refusals"] == 1 and s["wrong_refusals_n"] == 2
    assert s["avg_latency_s"] == 1.0           # errored row excluded from latency


def test_summarize_retrieval_only_has_no_generation_metrics():
    s = summarize([_row(hit=True, rr=0.5), _row(answerable=False)])
    assert s["refusal_correct"] is None and s["faithfulness"] is None

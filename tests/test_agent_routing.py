import pytest

from src import agent, llm

GOOD = [{"text": "A stack is LIFO.", "source": "ds.pdf", "page": 3, "score": 0.9, "rank": 1}]
WEAK = [{"text": "unrelated", "source": "ds.pdf", "page": 9, "score": 0.01, "rank": 1}]


class FakeRetriever:
    """Returns scripted result lists, one per search call (last one repeats)."""
    def __init__(self, *results):
        self.results, self.queries = list(results), []

    def search(self, query):
        self.queries.append(query)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class FakeChat:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        r = self.replies.pop(0) if self.replies else "fallback reply"
        if isinstance(r, Exception):
            raise r
        return r


def run(retriever, chat, question="What is a stack?", history=None, **kw):
    a = agent.build_agent(retriever, chat=chat, threshold=0.5, max_attempts=3, history_turns=4, **kw)
    return agent.run_agent(a, question, history)


@pytest.mark.parametrize("msg", ["hi", "Hello!", "thanks", "thank you!", "bye"])
def test_greeting_goes_to_direct_reply(msg):
    r, c = FakeRetriever(GOOD), FakeChat()
    out = run(r, c, msg)
    assert r.queries == [] and c.prompts == []
    assert out["answer"] and out["sources"] == []


def test_strong_score_generates_with_one_llm_call():
    r, c = FakeRetriever(GOOD), FakeChat("A stack is LIFO (ds.pdf, p. 3).")
    out = run(r, c)
    assert out["answer"].startswith("A stack")
    assert out["llm_calls"] == 1 and r.queries == ["What is a stack?"]
    assert out["sources"] == [{"source": "ds.pdf", "page": 3, "score": 0.9}]


def test_low_score_retries_then_generates():
    r = FakeRetriever(WEAK, GOOD)
    c = FakeChat("stack data structure LIFO", "Answer: A stack is LIFO.")
    out = run(r, c)
    assert len(r.queries) == 2 and r.queries[1] == "stack data structure LIFO"
    assert out["attempts"] == 2 and out["llm_calls"] == 2
    assert out["answer"] == "A stack is LIFO."


def test_three_weak_attempts_end_in_refusal_without_generating():
    r = FakeRetriever(WEAK)
    c = FakeChat("query two", "query three")
    out = run(r, c)
    assert len(r.queries) == 3
    assert out["answer"] == agent.REFUSAL and out["sources"] == []
    assert out["llm_calls"] == 2           # two rewrites, no generation call
    assert out["trace"][-1]["step"] == "refuse"


def test_follow_up_rewrites_with_history_first():
    r, c = FakeRetriever(GOOD), FakeChat("stack underflow definition", "Underflow is popping an empty stack.")
    hist = [{"role": "user", "content": "What is a stack?"}, {"role": "assistant", "content": "A LIFO structure."}]
    out = run(r, c, "what about underflow?", hist)
    assert r.queries == ["stack underflow definition"]
    assert "What is a stack?" in c.prompts[0]
    assert out["llm_calls"] == 2


def test_rewrite_failure_falls_back_to_keyword_query():
    r = FakeRetriever(WEAK, GOOD)
    c = FakeChat(llm.LLMError("down"), "ok")
    out = run(r, c, "What is the definition of a stack?")
    assert r.queries[1] == "definition stack"
    assert out["answer"] == "ok"


def test_model_refusal_drops_sources():
    r, c = FakeRetriever(GOOD), FakeChat(agent.REFUSAL)
    out = run(r, c)
    assert out["sources"] == []


def test_generation_error_propagates():
    r, c = FakeRetriever(GOOD), FakeChat(llm.LLMError("402"))
    with pytest.raises(llm.LLMError):
        run(r, c)


def test_best_threshold_separates_groups():
    from src.calibrate import best_threshold
    r = best_threshold([0.6, 0.8, 0.95], [0.0, 0.05, 0.2])
    assert 0.2 < r["threshold"] < 0.6
    assert r["balanced_accuracy"] == 1.0


def test_best_threshold_overlap_picks_best_compromise():
    from src.calibrate import best_threshold
    r = best_threshold([0.1, 0.7, 0.8, 0.9], [0.0, 0.02, 0.05, 0.75])
    assert r["answerable_pass_rate"] >= 0.75 and r["unanswerable_refusal_rate"] >= 0.75

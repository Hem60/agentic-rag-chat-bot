from src import evaluate


def _judge(monkeypatch, reply):
    monkeypatch.setattr(evaluate, "call_judge_inference", lambda prompt: reply)
    return evaluate.judge_answer("q", "expected", "generated", "ctx")


def test_clean_json(monkeypatch):
    r = _judge(monkeypatch, '{"score": 4, "reasoning": "fine"}')
    assert r == {"score": 4, "reasoning": "fine", "parse_failed": False}


def test_malformed_json_recovers_score_by_regex(monkeypatch):
    r = _judge(monkeypatch, '{"score": 5, "reasoning": "unterminated')
    assert r["score"] == 5 and r["parse_failed"]


def test_unparseable_reply_is_unscored_not_one(monkeypatch):
    r = _judge(monkeypatch, "I think the answer is good.")
    assert r["score"] is None and r["parse_failed"]


def test_missing_score_key_is_unscored(monkeypatch):
    r = _judge(monkeypatch, '{"reasoning": "no score"}')
    assert r["score"] is None

import pytest
from fastapi.testclient import TestClient

from src import api as api_mod
from src import llm


class _FakeAgent:
    def __init__(self, state=None, error=None):
        self.state, self.error, self.calls = state, error, []

    def invoke(self, inputs):
        self.calls.append(inputs)
        if self.error:
            raise self.error
        return self.state


GOOD_STATE = {
    "answer": "A stack is LIFO.",
    "sources": [{"source": "ds.pdf", "page": 3, "score": 0.9}],
    "chunks": [{"source": "ds.pdf", "page": 3, "score": 0.9, "text": "A stack is LIFO. " * 40}],
    "trace": [{"step": "route", "detail": "search"}],
    "best_score": 0.9, "attempts": 1, "llm_calls": 1,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(api_mod.app_state, "index_loaded", True)
    monkeypatch.setitem(api_mod.app_state, "reranker_loaded", True)
    monkeypatch.setitem(api_mod.app_state, "agent", _FakeAgent(GOOD_STATE))
    monkeypatch.setattr(api_mod, "limiter", api_mod.RateLimiter(100))
    return TestClient(api_mod.api)  # no `with`: skip the lifespan model loading


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["index_loaded"] is True and r.json()["reranker_loaded"] is True


def test_chat_returns_sources_with_pages_and_chunks(client):
    r = client.post("/chat", json={"question": "What is a stack?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "A stack is LIFO."
    assert body["sources"] == [{"source": "ds.pdf", "page": 3, "score": 0.9}]
    assert body["chunks"][0]["page"] == 3 and len(body["chunks"][0]["preview"]) <= 300
    assert body["trace"] and body["llm_calls"] == 1 and "latency_s" in body


def test_history_is_passed_to_agent(client):
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    client.post("/chat", json={"question": "and stacks?", "history": hist})
    sent = api_mod.app_state["agent"].calls[0]
    assert sent["history"] == hist and sent["question"] == "and stacks?"


def test_llm_error_returns_safe_503(client, monkeypatch):
    monkeypatch.setitem(api_mod.app_state, "agent", _FakeAgent(error=llm.LLMError("HTTP 402 secret-token")))
    r = client.post("/chat", json={"question": "What is a stack?"})
    assert r.status_code == 503 and "402" not in r.text and "token" not in r.text


def test_unexpected_error_does_not_leak_details(client, monkeypatch):
    monkeypatch.setitem(api_mod.app_state, "agent", _FakeAgent(error=ValueError("/secret/path boom")))
    r = client.post("/chat", json={"question": "What is a stack?"})
    assert r.status_code == 500 and "secret" not in r.text and "boom" not in r.text


def test_not_loaded_returns_503(client, monkeypatch):
    monkeypatch.setitem(api_mod.app_state, "index_loaded", False)
    assert client.post("/chat", json={"question": "What is a stack?"}).status_code == 503


def test_validation_rejects_bad_input(client):
    assert client.post("/chat", json={"question": "x"}).status_code == 422
    assert client.post("/chat", json={"question": "a" * 1001}).status_code == 422
    assert client.post("/chat", json={"question": "ok?", "history": [{"role": "system", "content": "x"}]}
                       ).status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api_mod, "limiter", api_mod.RateLimiter(2))
    codes = [client.post("/chat", json={"question": "What is a stack?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_window_expires():
    t = [0.0]
    rl = api_mod.RateLimiter(1, window_s=60, clock=lambda: t[0])
    assert rl.allow("a") and not rl.allow("a") and rl.allow("b")
    t[0] = 61
    assert rl.allow("a")


def test_cors_not_wildcard(client):
    r = client.options("/chat", headers={"Origin": "https://evil.example",
                                         "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in r.headers

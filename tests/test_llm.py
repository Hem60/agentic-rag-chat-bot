import pytest

from src import config, llm


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _HTTPError(Exception):
    def __init__(self, status):
        self.response = type("R", (), {"status_code": status})()


class _Client:
    """Fake InferenceClient: pops one scripted outcome per call."""
    def __init__(self, outcomes):
        self.outcomes, self.models = list(outcomes), []

    def chat_completion(self, model, **_):
        self.models.append(model)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _Resp(out)


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", "hf_test")


def _run(client, sleeps, **kw):
    return llm.chat([{"role": "user", "content": "hi"}], model="main",
                    client_factory=lambda: client, sleep=sleeps.append, **kw)


def test_retries_with_backoff_then_succeeds():
    sleeps = []
    client = _Client([_HTTPError(429), _HTTPError(503), "ok"])
    assert _run(client, sleeps, fallback_model="") == "ok"
    assert sleeps == [2, 4]


def test_gives_up_after_three_retries():
    sleeps = []
    client = _Client([_HTTPError(500)] * 4)
    with pytest.raises(llm.LLMError):
        _run(client, sleeps, fallback_model="")
    assert sleeps == [2, 4, 8]


def test_unserved_model_switches_to_fallback_without_waiting():
    sleeps = []
    client = _Client([_HTTPError(404), "from fallback"])
    assert _run(client, sleeps, fallback_model="backup") == "from fallback"
    assert client.models == ["main", "backup"] and sleeps == []


@pytest.mark.parametrize("status", [401, 402, 403])
def test_account_errors_fail_fast(status):
    sleeps = []
    client = _Client([_HTTPError(status), "never reached"])
    with pytest.raises(llm.LLMError) as exc:
        _run(client, sleeps, fallback_model="backup")
    assert client.models == ["main"] and sleeps == []
    assert str(status) in str(exc.value)

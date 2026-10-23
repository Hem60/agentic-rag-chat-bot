# -*- coding: utf-8 -*-
"""
src/llm.py
----------
Hugging Face chat-completion wrapper with retry, exponential backoff and a
fallback model.

Failure handling:
  * 429 / 5xx / timeouts / connection errors  -> retry (waits 2, 4, 8 s)
  * 404 / 400 (model not served here)         -> switch to the fallback model
  * 401 / 402 / 403 (auth, credits exhausted) -> fail immediately; retrying or
    switching model cannot help because the account itself is the problem
"""

import logging
import time
from typing import Callable, Optional

from src import config

log = logging.getLogger("rag.llm")


class LLMError(RuntimeError):
    """Raised when generation fails for good. The message is safe to show users."""


# Statuses that mean "the account/token is the problem" -- never retry these.
_FATAL_STATUS = {401, 402, 403}
# Statuses that mean "this model is not available" -- try the fallback model.
_MODEL_STATUS = {400, 404, 422}


def _status_of(exc: Exception) -> Optional[int]:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _default_client_factory():
    from huggingface_hub import InferenceClient
    return InferenceClient(token=config.HF_TOKEN, timeout=60)


def chat(
    messages: list,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    fallback_model: Optional[str] = None,
    client_factory: Callable = _default_client_factory,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Return the assistant reply text, or raise LLMError."""
    model = model or config.GENERATION_MODEL
    fallback = config.FALLBACK_MODEL if fallback_model is None else fallback_model
    max_tokens = max_tokens or config.LLM_MAX_TOKENS
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    if not config.token_is_set():
        raise LLMError("HUGGINGFACEHUB_API_TOKEN is not set.")

    client = client_factory()
    candidates = [model] + ([fallback] if fallback and fallback != model else [])
    last_err: Optional[Exception] = None

    for current in candidates:
        waits = list(config.LLM_RETRY_WAITS)
        while True:
            try:
                resp = client.chat_completion(
                    model=current,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - classified below
                last_err = e
                status = _status_of(e)
                if status in _FATAL_STATUS:
                    log.error("LLM fatal HTTP %s for %s", status, current)
                    raise LLMError(
                        f"Generation service refused the request (HTTP {status}). "
                        "Check the Hugging Face token and remaining free credits."
                    ) from e
                if status in _MODEL_STATUS:
                    log.warning("Model %s not served (HTTP %s); trying fallback", current, status)
                    break  # next candidate model
                if not waits:
                    log.warning("Retries exhausted for %s: %s", current, type(e).__name__)
                    break
                delay = waits.pop(0)
                log.info("LLM call failed (%s); retrying in %ss", status or type(e).__name__, delay)
                sleep(delay)

    raise LLMError("Generation service is unavailable right now.") from last_err

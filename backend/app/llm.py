"""LLM provider abstraction.

Default provider is Google Gemini (free tier) via plain httpx REST —
no ``google-genai`` SDK dependency. Fallback is any Ollama-compatible
OpenAI-style ``/chat/completions`` endpoint. With neither configured,
:func:`get_llm_client` returns :class:`NoLLMClient` and generation
falls back to clearly-labelled extractive answers (never faked).

Every ``generate`` call returns::

    {"text": str, "provider": str, "model": str,
     "usage": {"prompt_tokens": int|None, "completion_tokens": int|None,
               "total_tokens": int|None,
               "prompt_tokens_est": int, "completion_tokens_est": int,
               "total_tokens_est": int}}

Token counts are the real ``usage_metadata`` / ``usage`` values when
the API returns them; the ``*_est`` fields are always the chars/4
approximation so traces stay comparable across providers.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.embeddings import estimate_tokens

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _usage(prompt_text: str, completion_text: str, real: dict | None = None) -> dict:
    real = real or {}
    prompt_est = estimate_tokens(prompt_text)
    completion_est = estimate_tokens(completion_text)
    return {
        "prompt_tokens": real.get("prompt_tokens"),
        "completion_tokens": real.get("completion_tokens"),
        "total_tokens": real.get("total_tokens"),
        "prompt_tokens_est": prompt_est,
        "completion_tokens_est": completion_est,
        "total_tokens_est": (real.get("total_tokens")
                             or (prompt_est + completion_est)),
    }


class GeminiClient:
    """Gemini via REST: POST /v1beta/models/{model}:generateContent."""

    provider = "gemini"

    # Free-tier rate limits are tight (gemini-2.5-flash: ~10 req/min, which
    # is why the default model is gemini-3.5-flash-lite, ~30 req/min):
    # retry 429/503 with backoff instead of failing into the extractive
    # fallback on the first burst.
    _RETRYABLE = (429, 500, 502, 503, 504)
    _MAX_ATTEMPTS = 4
    _BACKOFF_SECONDS = (2.0, 5.0, 12.0)

    def _post_with_retry(self, url: str, **kwargs):
        import time

        last_exc = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                resp = self._client.post(url, **kwargs)
            except Exception as exc:  # transport error: retry like a 503
                last_exc = exc
                resp = None
            if resp is not None and resp.status_code not in self._RETRYABLE:
                resp.raise_for_status()
                return resp
            if attempt == self._MAX_ATTEMPTS:
                if resp is not None:
                    resp.raise_for_status()
                raise last_exc  # type: ignore[misc]
            retry_after = None
            if resp is not None:
                retry_after = resp.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else self._BACKOFF_SECONDS[attempt - 1]
            except ValueError:
                wait = self._BACKOFF_SECONDS[attempt - 1]
            wait = min(max(wait, 1.0), 60.0)
            logger.warning("Gemini %s (attempt %d/%d); retrying in %.0fs",
                           resp.status_code if resp is not None else "transport error",
                           attempt, self._MAX_ATTEMPTS, wait)
            time.sleep(wait)
        raise last_exc  # pragma: no cover - loop always returns/raises above

    def __init__(self, api_key: str, model: str):
        from app.http import make_client

        # Proxy handling lives in app.http: httpx's own env parsing
        # breaks on this machine's no_proxy value, and trust_env=False
        # breaks the opposite way (no direct internet here).
        self._client = make_client(60.0, base_url=GEMINI_API_BASE)
        self._api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str | None = None) -> dict:
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        contents: list[dict[str, Any]] = []
        if system:
            contents.append({"role": "user",
                             "parts": [{"text": f"System instruction: {system}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        resp = self._post_with_retry(
            url,
            headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            json={"contents": contents,
                  "generationConfig": {"maxOutputTokens": settings.MAX_OUTPUT_TOKENS}},
        )
        data = resp.json()
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content", {}).get("parts", [])
                 if candidates else [])
        text = "".join(p.get("text", "") for p in parts).strip()
        meta = data.get("usageMetadata") or {}
        real = {
            "prompt_tokens": meta.get("promptTokenCount"),
            "completion_tokens": meta.get("candidatesTokenCount"),
            "total_tokens": meta.get("totalTokenCount"),
        }
        return {"text": text, "provider": self.provider, "model": self.model,
                "usage": _usage(prompt, text, real)}

    def generate_stream(self, prompt: str, system: str | None = None):
        """Stream Gemini via ``:streamGenerateContent?alt=sse``.

        Yields ``{"delta": str}`` per text chunk, then a final
        ``{"done": True, "text": str, "usage": {...}}`` event. A retryable
        HTTP status before the first chunk is retried (same backoff as
        the blocking call); once chunks have flowed, errors propagate.
        """
        import json
        import time

        url = f"{GEMINI_API_BASE}/models/{self.model}:streamGenerateContent?alt=sse"
        contents: list[dict[str, Any]] = []
        if system:
            contents.append({"role": "user",
                             "parts": [{"text": f"System instruction: {system}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        body = {"contents": contents,
                "generationConfig": {"maxOutputTokens": settings.MAX_OUTPUT_TOKENS}}

        full_parts: list[str] = []
        real: dict[str, Any] = {}
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                with self._client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code in self._RETRYABLE and not full_parts:
                        if attempt == self._MAX_ATTEMPTS:
                            resp.raise_for_status()
                        wait = self._BACKOFF_SECONDS[attempt - 1]
                        logger.warning("Gemini stream %s (attempt %d/%d); retrying in %.0fs",
                                       resp.status_code, attempt, self._MAX_ATTEMPTS, wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            data = json.loads(payload)
                        except ValueError:
                            continue
                        meta = data.get("usageMetadata") or {}
                        if meta:
                            real = {
                                "prompt_tokens": meta.get("promptTokenCount"),
                                "completion_tokens": meta.get("candidatesTokenCount"),
                                "total_tokens": meta.get("totalTokenCount"),
                            }
                        candidates = data.get("candidates") or []
                        parts = (candidates[0].get("content", {}).get("parts", [])
                                 if candidates else [])
                        delta = "".join(p.get("text", "") for p in parts)
                        if delta:
                            full_parts.append(delta)
                            yield {"delta": delta}
                    break
            except Exception:
                if full_parts:
                    raise
                if attempt == self._MAX_ATTEMPTS:
                    raise
                wait = self._BACKOFF_SECONDS[attempt - 1]
                logger.warning("Gemini stream transport error (attempt %d/%d); retrying in %.0fs",
                               attempt, self._MAX_ATTEMPTS, wait)
                time.sleep(wait)
        text = "".join(full_parts).strip()
        yield {"done": True, "text": text, "usage": _usage(prompt, text, real)}


class OllamaClient:
    """Any OpenAI-compatible /chat/completions endpoint (Ollama default)."""

    provider = "ollama"

    def __init__(self, base_url: str, api_key: str, model: str):
        from app.http import make_client

        # Proxy handling lives in app.http (localhost targets go direct).
        self._client = make_client(120.0, base_url=base_url)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str | None = None) -> dict:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages, "temperature": 0.2,
                  "max_tokens": settings.MAX_OUTPUT_TOKENS},
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        real = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        return {"text": (text or "").strip(), "provider": self.provider,
                "model": self.model, "usage": _usage(prompt, text or "", real)}

    def generate_stream(self, prompt: str, system: str | None = None):
        """Stream Ollama/OpenAI-compatible ``/chat/completions`` (``stream: true``).

        Yields ``{"delta": str}`` per content chunk, then a final
        ``{"done": True, "text": str, "usage": {...}}`` event. NDJSON lines
        carry ``choices[0].delta.content``; the final ``done`` line carries
        ``prompt_eval_count`` / ``eval_count`` when the server reports them.
        """
        import json

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        full_parts: list[str] = []
        real: dict[str, Any] = {}
        with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages,
                  "temperature": 0.2, "stream": True,
                  "max_tokens": settings.MAX_OUTPUT_TOKENS},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                text_line = line[5:].strip() if line.startswith("data:") else line.strip()
                if not text_line or text_line == "[DONE]":
                    continue
                try:
                    data = json.loads(text_line)
                except ValueError:
                    continue
                if data.get("done") is True:
                    if data.get("prompt_eval_count") is not None:
                        real = {
                            "prompt_tokens": data.get("prompt_eval_count"),
                            "completion_tokens": data.get("eval_count"),
                            "total_tokens": ((data.get("prompt_eval_count") or 0)
                                             + (data.get("eval_count") or 0)),
                        }
                    continue
                delta = ((data.get("choices") or [{}])[0].get("delta", {})
                         .get("content", ""))
                if delta:
                    full_parts.append(delta)
                    yield {"delta": delta}
                usage = data.get("usage") or {}
                if usage:
                    real = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
        text = "".join(full_parts).strip()
        yield {"done": True, "text": text, "usage": _usage(prompt, text, real)}


class NoLLMClient:
    """No LLM configured — generate() raises so callers use the extractive fallback."""

    provider = "none"
    model = ""

    def generate(self, prompt: str, system: str | None = None) -> dict:
        raise RuntimeError(
            "No LLM configured: set GEMINI_API_KEY or LLM_BASE_URL. "
            "Extractive fallback will be used instead."
        )

    def generate_stream(self, prompt: str, system: str | None = None):
        raise RuntimeError(
            "No LLM configured: set GEMINI_API_KEY or LLM_BASE_URL. "
            "Extractive fallback will be used instead."
        )


_llm_client = None


def get_llm_client():
    """Cached provider client selected only from env vars (see config).

    If the chosen provider's transport (httpx) is not installed, degrade
    to NoLLMClient instead of crashing — the extractive fallback applies.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    provider = settings.effective_llm_provider
    try:
        if provider == "gemini":
            _llm_client = GeminiClient(settings.GEMINI_API_KEY, settings.GEMINI_MODEL)
            logger.info("LLM: Gemini %s", settings.GEMINI_MODEL)
        elif provider == "ollama":
            _llm_client = OllamaClient(
                settings.LLM_BASE_URL, settings.LLM_API_KEY or settings.OPENAI_API_KEY,
                settings.LLM_MODEL)
            logger.info("LLM: Ollama-compatible %s at %s", settings.LLM_MODEL, settings.LLM_BASE_URL)
        else:
            raise RuntimeError("no LLM provider configured")
    except Exception as exc:
        logger.warning("LLM: %s provider unavailable (%s) — extractive fallback only.", provider, exc)
        _llm_client = NoLLMClient()
    if isinstance(_llm_client, NoLLMClient):
        logger.info("LLM: none configured — extractive fallback only (labelled as such).")
    return _llm_client


def llm_available() -> bool:
    return not isinstance(get_llm_client(), NoLLMClient)

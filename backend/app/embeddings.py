"""Embeddings.

Provider order (see ``app/config.py``):
  1. Gemini ``gemini-embedding-001`` (768-dim) when ``GEMINI_API_KEY`` is
     set and ``EMBEDDING_PROVIDER`` is ``gemini``/``auto`` — the default.
  2. sentence-transformers ``all-MiniLM-L6-v2`` (384-dim) — local fallback.
  3. Deterministic hash-based embedding (384-dim) — dev fallback so the
     API still works locally with zero setup / no model download. It
     captures keyword overlap via hashing, not semantics, and is
     clearly logged when active.

The embedding dimension is ``settings.embedding_dim`` (768 for Gemini,
384 for MiniLM/hash, or an explicit ``EMBEDDING_DIM`` override) and the
Qdrant collection is created with exactly that dimension.
"""
from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

DIM_LOCAL = 384
DIM_GEMINI = 768
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return max(0, len(text) // 4)


def _hash_embed(text: str, dim: int = DIM_LOCAL) -> list[float]:
    """Deterministic bag-of-hashed-tokens embedding, L2-normalised."""
    vec = np.zeros(dim, dtype=np.float64)
    for token in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(float).tolist()


class HashEmbeddingFunction:
    """Dev fallback embedding function (LangChain-compatible interface)."""

    dim = DIM_LOCAL
    name = "hash-fallback-384"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_embed(text, self.dim)


class SentenceTransformerEmbeddingFunction:
    """sentence-transformers wrapper (LangChain-compatible interface)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=float).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=float).tolist()


class GeminiEmbeddingFunction:
    """Gemini embeddings via plain httpx REST (no google-genai SDK).

    POST https://generativelanguage.googleapis.com/v1beta/models/
    <model>:embedContent  with ``x-goog-api-key`` header.
    """

    def __init__(self, api_key: str, model: str, dim: int = DIM_GEMINI,
                 api_key_secondary: str = ""):
        from app.http import make_client

        # Proxy handling lives in app.http (see module docstring there).
        self._client = make_client(30.0, base_url="https://generativelanguage.googleapis.com")
        self._api_key = api_key
        self._api_key_secondary = api_key_secondary
        self._on_secondary = False
        self._model = model
        self.dim = dim
        self.name = f"gemini/{model}"

    def _switch_to_secondary(self) -> bool:
        """Switch to the secondary Gemini key, once (sticky).

        A key that hit its daily quota stays exhausted until it
        resets, so there is no point retrying the primary on the
        next call either. No-op when no secondary key is set or
        the switch already happened.
        """
        if not self._api_key_secondary or self._on_secondary:
            return False
        logger.warning("Gemini embed: primary key exhausted; switching to secondary key")
        self._api_key = self._api_key_secondary
        self._on_secondary = True
        return True

    def _post_with_retry(self, url: str, payload: dict):
        """POST with the retry policy shared by single and batch embeds.

        Retries 429/5xx with backoff (2s, 5s, 12s). A daily-quota 429
        whose retry-after is hours fails fast so the caller can switch
        keys instead of sleeping through the ladder. Returns the
        successful response, or raises the last error.
        """
        import time

        while True:
            headers = {"x-goog-api-key": self._api_key,
                       "Content-Type": "application/json"}
            last_exc = None
            resp = None
            for attempt in range(1, 5):
                try:
                    resp = self._client.post(url, headers=headers, json=payload)
                except Exception as exc:  # transport error: retry like a 503
                    last_exc = exc
                    resp = None
                if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                    break
                if attempt == 4:
                    break
                if resp is not None and resp.status_code == 429:
                    try:
                        ra = float(resp.headers.get("retry-after") or 0)
                    except ValueError:
                        ra = 0.0
                    if ra > 300:
                        logger.warning(
                            "Gemini embed quota exhausted (retry-after %.0fs); "
                            "failing fast instead of retrying.", ra)
                        break
                wait = 12.0 if attempt == 3 else (5.0 if attempt == 2 else 2.0)
                if resp is not None:
                    retry_after = resp.headers.get("retry-after")
                    try:
                        wait = float(retry_after) if retry_after else wait
                    except ValueError:
                        pass
                wait = min(max(wait, 1.0), 60.0)
                logger.warning("Gemini embed %s (attempt %d/4); retrying in %.0fs",
                               resp.status_code if resp is not None else "transport error",
                               attempt, wait)
                time.sleep(wait)
            if resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
                return resp  # success
            if resp is not None and resp.status_code == 429 and self._switch_to_secondary():
                continue
            if resp is not None:
                resp.raise_for_status()
            raise last_exc  # type: ignore[misc]

    def _normalize(self, values: list[float]) -> list[float]:
        """Pad/truncate to the configured dim and L2-normalise."""
        vec = np.asarray(values, dtype=float)
        if len(vec) != self.dim:
            out = np.zeros(self.dim, dtype=float)
            out[: min(self.dim, len(vec))] = vec[: min(self.dim, len(vec))]
            vec = out
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed_one(self, text: str) -> list[float]:
        # Free-tier embedding quota is tight and the evaluator + live
        # traffic can burst past it: _post_with_retry handles the
        # 429/5xx backoff and the secondary-key fallover.
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:embedContent"
        )
        payload = {"model": f"models/{self._model}",
                   "content": {"parts": [{"text": text}]},
                   # Ask for exactly the configured dim (768 by default):
                   # the model natively returns 3072 dims, and truncating
                   # that vector afterwards would throw away signal.
                   "outputDimensionality": self.dim}
        resp = self._post_with_retry(url, payload)
        return self._normalize(resp.json()["embedding"]["values"])

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed up to 100 texts in ONE batchEmbedContents call.

        This is what makes big ingests fast: the old one-call-per-chunk
        loop needed ~165 HTTP calls for a 46-page PDF (60s+ and the
        request died); one batch call covers 100 chunks.
        """
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents"
        )
        payload = {"requests": [
            {"model": f"models/{self._model}",
             "content": {"parts": [{"text": t}]},
             "outputDimensionality": self.dim}
            for t in texts
        ]}
        resp = self._post_with_retry(url, payload)
        embeddings = resp.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Batch embed returned {len(embeddings)} vectors for "
                f"{len(texts)} texts.")
        logger.info("Gemini batch embed: %d texts in 1 call", len(texts))
        return [self._normalize(e.get("values", [])) for e in embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # Gemini caps a batch at 100 requests; big ingests go through in
        # 100-text batches. A batch that fails falls back to one-by-one
        # for that batch only, so one bad chunk never loses the rest.
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            try:
                vectors.extend(self._embed_batch(batch))
            except Exception as exc:
                logger.warning("Batch embed failed (%s); embedding %d texts "
                               "one by one.", exc, len(batch))
                vectors.extend(self._embed_one(t) for t in batch)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


_embedding_fn = None


def get_embedding_function():
    """Return a cached embedding function per the configured provider."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    provider = settings.effective_embedding_provider
    if provider == "gemini":
        try:
            fn = GeminiEmbeddingFunction(
                settings.GEMINI_API_KEY, settings.GEMINI_EMBEDDING_MODEL,
                dim=settings.embedding_dim,
                api_key_secondary=settings.GEMINI_API_KEY_SECONDARY,
            )
            # Probe with a tiny call so a bad key fails fast, at startup,
            # instead of mid-retrieval; fall through to local on failure.
            fn.embed_query("connectivity probe")
            logger.info("Embeddings: Gemini %s (dim=%d)", fn.name, fn.dim)
            _embedding_fn = fn
            return _embedding_fn
        except Exception as exc:
            logger.warning("Embeddings: Gemini unavailable (%s); trying local.", exc)
    if provider in ("local", "gemini", "auto"):
        try:
            fn = SentenceTransformerEmbeddingFunction(settings.EMBEDDING_MODEL)
            logger.info("Embeddings: sentence-transformers %s (dim=%d)", fn.name, fn.dim)
            _embedding_fn = fn
            return _embedding_fn
        except Exception as exc:  # ImportError, model download failure, no torch, ...
            logger.warning(
                "Embeddings: sentence-transformers unavailable (%s); "
                "using deterministic hash fallback (384-dim, dev only).",
                exc,
            )
    _embedding_fn = HashEmbeddingFunction()
    logger.info("Embeddings: hash fallback (dim=%d)", _embedding_fn.dim)
    return _embedding_fn


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_function().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embedding_function().embed_query(text)


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity between two vectors (0.0 if either is zero)."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

"""Response cache: exact-match + semantic, Redis with in-process fallback.

Why two layers
-------------
* **Exact-match** — the same question asked again (same wording, same
  conversation context) must never touch retrieval or the LLM again.
  Normalized question + history hash -> SHA-256 key.
* **Semantic** — a close paraphrase should hit too. The question is
  embedded with the existing embedding function and compared (cosine)
  against stored question embeddings; above
  ``CACHE_SEMANTIC_THRESHOLD`` (default 0.95) it counts as a hit.

Backends (same degrade-gracefully pattern as the vector store)
------------------------------------------------------------
* ``redis`` — real Redis at ``REDIS_URL`` (default
  ``redis://localhost:6379``); entries are JSON with a TTL.
* ``memory`` — when Redis is unreachable: a bounded in-process
  ``OrderedDict`` (``CACHE_MAX_ENTRIES``, default 200), so the app still
  works with zero setup. Entries expire lazily by timestamp.

Correctness
-----------
* A hit is only valid while the documents the answer was grounded in
  still exist: each entry records the cited documents' ids (matched by
  title+source against the live corpus) and a corpus version. The
  lookup re-checks those ids against the live corpus; a deleted cited
  document turns the hit into a miss. ``invalidate_document`` also
  purges citing entries eagerly on ``DELETE /api/documents/{id}``.
* History-aware: the key includes a hash of the prior turns, so a
  follow-up like "tell me more" in one conversation can never hit the
  cache of the same words in another conversation. Semantic matches
  must share the same history hash.
* Role-aware (guardrails): the key also includes the requesting role.
  Without it, an answer an exec legitimately received (e.g. a comp
  figure from a restricted document) could later be served to an
  employee asking the same question — a privilege escalation through
  the cache. Entries stored before roles existed carry no role and
  are treated as employee entries (the safe direction).
* Never cached: errors, empty answers, confidence 0, the "clarify"
  route, and decline answers ("I don't know" / "couldn't find ...").
  Caching a decline would freeze a "no" that a later ingest could fix.

The cache is best-effort: any Redis failure degrades to the in-memory
store for that operation (logged), never to a 500.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict

from app.config import settings

logger = logging.getLogger(__name__)

EXACT_PREFIX = "rag:cache:exact:"
SEM_PREFIX = "rag:cache:sem:"
_ALL_PREFIXES = (EXACT_PREFIX, SEM_PREFIX)

# Answers starting with (or containing) these are declines, not answers.
# The LLM phrases a grounded "I don't know" many ways ("there is no
# information regarding ...", "the context does not contain ..."), so
# the list covers the common wordings; a legit answer essentially
# never contains these phrases.
DECLINE_MARKERS = (
    "couldn't find relevant information",
    "i don't know",
    "could you clarify",
    "no documents to answer from",
    "there is no information",
    "no information regarding",
    "no information about",
    "does not contain the answer",
    "does not contain information",
    "does not contain any information",
    "not mentioned in",
    "is not mentioned",
    "no mention of",
    "cannot be determined from",
    "not provided in",
    "i cannot answer",
    "unable to answer",
)

_WORD_RE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[.?!…\s]+$")


def _redact_url(url: str) -> str:
    """Hide credentials in a Redis URL for logs / /health output.

    ``rediss://default:<password>@host`` -> ``rediss://default:***@host``.
    A URL without a password is returned unchanged.
    """
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url or "")
        if parsed.password:
            host = parsed.netloc.rsplit("@", 1)[-1]
            user = parsed.username or ""
            netloc = f"{user}:***@{host}" if user else f"***@{host}"
            return urlunparse((parsed.scheme, netloc, parsed.path,
                               parsed.params, parsed.query, parsed.fragment))
    except Exception:  # noqa: BLE001 - redaction must never break startup
        pass
    return url or ""


def normalize_question(question: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation."""
    text = _WORD_RE.sub(" ", (question or "").strip().lower())
    return _TRAILING_PUNCT.sub("", text)


def history_key(history) -> str:
    """Hash of the prior turns ("" when there is no history).

    Only role + normalized content count, so formatting noise in the
    client payload cannot split the cache. The history agent already
    caps turns at MAX_HISTORY_TURNS pairs; we hash whatever arrives.
    """
    if not history:
        return ""
    turns = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip().lower()
        content = normalize_question(str(turn.get("content", "")))
        if content:
            turns.append(f"{role}:{content}")
    if not turns:
        return ""
    return hashlib.sha256("\n".join(turns).encode("utf-8")).hexdigest()


def _exact_key(question: str, history, role: str = "employee") -> str:
    raw = (f"exact|{normalize_question(question)}|{history_key(history)}|"
           f"role={(role or '').strip().lower() or 'employee'}")
    return EXACT_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sem_key(question: str, history, role: str = "employee") -> str:
    raw = (f"sem|{normalize_question(question)}|{history_key(history)}|"
           f"role={(role or '').strip().lower() or 'employee'}")
    return SEM_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def corpus_version(doc_ids) -> str:
    """Hash of the sorted live document ids — one version per corpus."""
    return hashlib.sha256("|".join(sorted(doc_ids)).encode("utf-8")).hexdigest()


def is_cacheable(result: dict | None) -> bool:
    """Only real, grounded answers are worth caching."""
    if not result or not isinstance(result, dict):
        return False
    answer = str(result.get("answer") or "").strip()
    if not answer:
        return False
    if float(result.get("confidence") or 0.0) <= 0:
        return False
    if str(result.get("route") or "") == "clarify":
        return False
    lowered = answer.lower()
    if any(marker in lowered for marker in DECLINE_MARKERS):
        return False
    return True


def _cited_doc_ids(sources, documents) -> list[str]:
    """Ids of the corpus documents a result cited (matched by title+source).

    Sources carry title/source but no document id, so match against the
    live document list. Anything unmatched falls back to the full corpus
    (conservative: a later delete of any doc invalidates the entry).
    """
    if not sources:
        return []
    pairs = {(str(s.get("title", "")), str(s.get("source", ""))) for s in sources}
    ids = [d["id"] for d in documents
           if (str(d.get("title", "")), str(d.get("source", ""))) in pairs]
    if len(ids) != len(pairs):
        return sorted({d["id"] for d in documents})
    return sorted(ids)


def _source_keys(sources) -> list[str]:
    return sorted({f"{s.get('title', '')}\x00{s.get('source', '')}"
                   for s in (sources or [])})


class ResponseCache:
    """Exact + semantic response cache over Redis (fallback: memory)."""

    def __init__(self) -> None:
        self.mode = "memory"
        self._client = None
        self._memory: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self.ttl = int(getattr(settings, "CACHE_TTL_SECONDS", 86400))
        self.threshold = float(getattr(settings, "CACHE_SEMANTIC_THRESHOLD", 0.95))
        self.max_entries = int(getattr(settings, "CACHE_MAX_ENTRIES", 200))
        self.sem_scan_max = int(getattr(settings, "CACHE_SEMANTIC_SCAN_MAX", 500))
        try:
            import redis

            client = redis.Redis.from_url(
                settings.REDIS_URL, socket_connect_timeout=2,
                socket_timeout=5, decode_responses=True)
            client.ping()
            self._client = client
            self.mode = "redis"
            logger.info("Response cache: Redis active (%s, ttl=%ds, "
                        "semantic threshold=%.2f)", _redact_url(settings.REDIS_URL),
                        self.ttl, self.threshold)
        except Exception as exc:
            logger.warning(
                "Response cache: Redis unreachable (%s); using bounded "
                "in-process cache (%d entries). Start Redis for a shared, "
                "persistent cache.", exc, self.max_entries)
            logger.info("Response cache: in-memory active (ttl=%ds, semantic "
                        "threshold=%.2f)", self.ttl, self.threshold)

    # -- introspection ----------------------------------------------------

    def status(self) -> dict:
        return {"cache": self.mode, "redis_url": _redact_url(settings.REDIS_URL),
                "ttl_seconds": self.ttl,
                "semantic_threshold": self.threshold,
                "entries": self._count_entries()}

    def _count_entries(self) -> int:
        if self._client is not None:
            try:
                n = 0
                for prefix in _ALL_PREFIXES:
                    for _ in self._client.scan_iter(match=f"{prefix}*",
                                                    count=100):
                        n += 1
                return n
            except Exception as exc:  # noqa: BLE001 - best-effort status
                logger.warning("Response cache: count failed (%s)", exc)
                return -1
        now = time.monotonic()
        return sum(1 for _, exp in self._memory.values() if exp > now)

    # -- low-level get/set ------------------------------------------------

    def _redis_get(self, key: str) -> dict | None:
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def _redis_set(self, key: str, entry: dict) -> None:
        self._client.set(key, json.dumps(entry), ex=self.ttl)

    def _redis_delete(self, key: str) -> None:
        self._client.delete(key)

    def _redis_scan(self, prefix: str, limit: int) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        for key in self._client.scan_iter(match=f"{prefix}*", count=100):
            if len(out) >= limit:
                break
            try:
                entry = self._redis_get(key)
            except Exception:  # noqa: BLE001 - one bad entry must not fail lookup
                continue
            if entry is not None:
                out.append((key, entry))
        return out

    def _memory_get(self, key: str) -> dict | None:
        item = self._memory.get(key)
        if item is None:
            return None
        entry, expires = item
        if expires <= time.monotonic():
            del self._memory[key]
            return None
        self._memory.move_to_end(key)
        return entry

    def _memory_set(self, key: str, entry: dict) -> None:
        self._memory[key] = (entry, time.monotonic() + self.ttl)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def _memory_delete(self, key: str) -> None:
        self._memory.pop(key, None)

    def _get(self, key: str) -> dict | None:
        if self._client is not None:
            try:
                return self._redis_get(key)
            except Exception as exc:  # noqa: BLE001 - degrade to memory
                logger.warning("Response cache: Redis get failed (%s); "
                               "using in-memory store for this call.", exc)
        return self._memory_get(key)

    def _set(self, key: str, entry: dict) -> None:
        if self._client is not None:
            try:
                self._redis_set(key, entry)
                return
            except Exception as exc:  # noqa: BLE001 - degrade to memory
                logger.warning("Response cache: Redis set failed (%s); "
                               "using in-memory store for this call.", exc)
        self._memory_set(key, entry)

    def _delete(self, key: str) -> None:
        if self._client is not None:
            try:
                self._redis_delete(key)
            except Exception as exc:  # noqa: BLE001 - best-effort delete
                logger.warning("Response cache: Redis delete failed (%s)", exc)
        self._memory_delete(key)

    def _scan(self, prefix: str, limit: int) -> list[tuple[str, dict]]:
        if self._client is not None:
            try:
                return self._redis_scan(prefix, limit)
            except Exception as exc:  # noqa: BLE001 - degrade to memory
                logger.warning("Response cache: Redis scan failed (%s); "
                               "using in-memory store for this call.", exc)
        now = time.monotonic()
        out = []
        for key, (entry, expires) in list(self._memory.items()):
            if not key.startswith(prefix):
                continue
            if expires <= now:
                del self._memory[key]
                continue
            out.append((key, entry))
            if len(out) >= limit:
                break
        return out

    # -- validity ---------------------------------------------------------

    @staticmethod
    def _valid(entry: dict, current_doc_ids: set[str]) -> bool:
        """Cited documents must all still exist in the live corpus."""
        doc_ids = entry.get("doc_ids") or []
        if not doc_ids:
            return True  # nothing cited (e.g. a greeting) — nothing to go stale
        return set(doc_ids).issubset(current_doc_ids)

    # -- public API -------------------------------------------------------

    def lookup(self, question: str, history, documents: list[dict],
               role: str = "employee"
               ) -> tuple[dict | None, str | None, list[float] | None,
                          float | None]:
        """(entry, hit_type, query_embedding, similarity) — entry None on a miss.

        Exact match first (no embedding call); semantic second (one
        embedding call, skipped entirely on an exact hit). ``similarity``
        is the cosine score for semantic hits, else None. ``role``
        scopes the lookup: an entry stored for another role never hits
        (see the module docstring — without this, cached answers would
        leak across roles).
        """
        role = (role or "").strip().lower() or "employee"
        current_ids = {d["id"] for d in documents}
        hkey = history_key(history)

        exact = self._get(_exact_key(question, history, role))
        if exact is not None:
            if self._valid(exact, current_ids):
                logger.info("Response cache: exact hit (%s)",
                            normalize_question(question)[:60])
                return exact, "exact", None, None
            self._delete(_exact_key(question, history, role))
            logger.info("Response cache: exact entry stale (cited doc gone); "
                        "treating as a miss.")

        try:
            from app.embeddings import cosine_similarity, get_embedding_function

            query_vec = get_embedding_function().embed_query(question)
        except Exception as exc:  # noqa: BLE001 - embeddings must not break chat
            logger.warning("Response cache: embedding failed (%s); semantic "
                           "lookup skipped.", exc)
            return None, None, None, None

        best: dict | None = None
        best_score = 0.0
        best_key: str | None = None
        for key, entry in self._scan(SEM_PREFIX, self.sem_scan_max):
            if entry.get("history_key", "") != hkey:
                continue
            # Role-scoped: entries stored for another role never match.
            # Legacy entries (stored before roles) carry no role and are
            # treated as employee entries.
            if (entry.get("role") or "employee") != role:
                continue
            vec = entry.get("embedding")
            if not vec:
                continue
            try:
                score = cosine_similarity(query_vec, vec)
            except Exception:  # noqa: BLE001 - one bad vector must not fail lookup
                continue
            if score > best_score:
                best, best_score, best_key = entry, score, key
        if best is not None and best_score >= self.threshold:
            if self._valid(best, current_ids):
                logger.info("Response cache: semantic hit (similarity %.3f) "
                            "for '%s'", best_score,
                            normalize_question(question)[:60])
                return best, "semantic", query_vec, round(best_score, 3)
            if best_key:
                self._delete(best_key)
            logger.info("Response cache: semantic entry stale (cited doc "
                        "gone); treating as a miss.")
        return None, None, query_vec, None

    def store(self, result: dict, question: str, history,
              documents: list[dict],
              query_embedding: list[float] | None = None,
              role: str = "employee") -> bool:
        """Cache a fresh result under both its exact and semantic keys."""
        if not is_cacheable(result):
            return False
        role = (role or "").strip().lower() or "employee"
        current_ids = {d["id"] for d in documents}
        entry = {
            "answer": result.get("answer", ""),
            "confidence": float(result.get("confidence") or 0.0),
            "sources": result.get("sources") or [],
            "route": result.get("route", ""),
            "retries": int(result.get("retries") or 0),
            "run_id": result.get("run_id", ""),
            "trace": result.get("trace") or {},
            "question": normalize_question(question),
            "history_key": history_key(history),
            "role": role,
            "doc_ids": _cited_doc_ids(result.get("sources") or [], documents),
            "source_keys": _source_keys(result.get("sources") or []),
            "corpus_version": corpus_version(current_ids),
            "created_at": time.time(),
        }
        self._set(_exact_key(question, history, role), entry)
        if query_embedding is None:
            try:
                from app.embeddings import get_embedding_function

                query_embedding = get_embedding_function().embed_query(question)
            except Exception as exc:  # noqa: BLE001 - semantic entry is optional
                logger.warning("Response cache: embedding failed (%s); "
                               "storing exact-match entry only.", exc)
                return True
        sem_entry = dict(entry)
        sem_entry["embedding"] = [float(v) for v in query_embedding]
        self._set(_sem_key(question, history, role), sem_entry)
        return True

    def invalidate_document(self, doc_id: str, title: str = "",
                            source: str = "") -> int:
        """Drop every cached entry that cited this document. Returns count."""
        target_key = f"{title}\x00{source}" if (title or source) else ""
        dropped = 0
        for prefix in _ALL_PREFIXES:
            for key, entry in self._scan(prefix, self.sem_scan_max):
                cited = entry.get("source_keys") or []
                if doc_id in (entry.get("doc_ids") or []) or \
                        (target_key and target_key in cited):
                    self._delete(key)
                    dropped += 1
        if dropped:
            logger.info("Response cache: invalidated %d entries citing "
                        "document %s", dropped, doc_id)
        return dropped


def build_cached_trace(question: str, run_id: str, corpus_stats: dict | None,
                       entry: dict, hit_type: str, serve_ms: float,
                       similarity: float | None = None) -> dict:
    """Trace for a cache-served answer: one cache_hit step, zero LLM calls.

    Mirrors the shape of graph._new_trace so the frontend's trace
    normalizer and stats derivation work unchanged. Retrieved/kept
    counts come from the original run (the answer is still grounded in
    those chunks); llm_calls is empty and total tokens 0, because this
    serving made no LLM call and ran no retrieval.
    """
    orig_trace = entry.get("trace") or {}
    step = {
        "node": "cache_hit", "duration_ms": serve_ms,
        "match_type": hit_type,
        "cached_from_run_id": entry.get("run_id", ""),
        "detail": (f"Served from cache ({hit_type} match) — 0 LLM calls, "
                   f"no retrieval"),
    }
    if similarity is not None:
        step["similarity"] = round(similarity, 3)
    return {
        "run_id": run_id, "question": question,
        "corpus": corpus_stats or {"documents_ingested": 0, "total_chars": 0,
                                   "total_tokens_est": 0, "total_chunks": 0},
        "steps": [step], "timings_ms": {"total": serve_ms, "cache_hit": serve_ms},
        "llm_calls": [], "total_llm_tokens": 0, "total_llm_tokens_est": 0,
        "retrieved": int(orig_trace.get("retrieved") or 0),
        "kept": int(orig_trace.get("kept") or 0),
        "discarded": 0,
        "rewritten_question": None, "route": entry.get("route", ""),
        "retries": 0, "contextualized_question": None, "history_turns": 0,
        "context_budget": {},
        "cached": True, "cache_hit_type": hit_type,
        "cached_from_run_id": entry.get("run_id", ""),
    }


_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    """Process-wide singleton (mode is fixed at first use, like the store)."""
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache

"""Tavily web search — the optional fallback retrieval path.

Classic Adaptive RAG pattern: when local documents cannot answer a
question, search the web and answer from those results, with web
citations. Uses the tavily-python SDK when it is installed, else a raw
httpx POST to https://api.tavily.com/search.

Every failure mode — no key, bad key, network error, rate limit,
quota exhausted — degrades to "no web results". This module never
raises into a chat run and never logs the API key.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_VALID_DEPTHS = ("basic", "advanced")


def web_search_available() -> bool:
    """True only when the fallback is switched on AND a key is set."""
    return bool(settings.web_search_enabled)


def _normalize_results(results: Any) -> list[dict]:
    """Tavily result dicts -> retrieval chunks, same shape as local docs
    (title / text / source / score) plus source_type="web"."""
    docs: list[dict] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        content = (r.get("content") or "").strip()
        url = (r.get("url") or "").strip()
        if not content or not url:
            continue
        try:
            score = float(r.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        docs.append({
            "title": (r.get("title") or url).strip(),
            "text": content,
            "source": url,
            "score": max(0.0, min(1.0, score)),
            "source_type": "web",
        })
    return docs


def _search_via_sdk(query: str, max_results: int,
                    search_depth: str) -> list[dict]:
    """tavily-python SDK path. Raises ImportError when the SDK is absent
    (caller falls back to raw httpx); any other failure propagates to
    the caller's graceful-degradation handler."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    resp = client.search(query=query, search_depth=search_depth,
                         max_results=max_results)
    return (resp or {}).get("results") or []


def _search_via_http(query: str, max_results: int,
                     search_depth: str) -> list[dict]:
    """Raw httpx path (no SDK dependency)."""
    import httpx

    resp = httpx.post(TAVILY_SEARCH_URL, json={
        "api_key": settings.TAVILY_API_KEY, "query": query,
        "search_depth": search_depth, "max_results": max_results,
    }, timeout=30.0)
    resp.raise_for_status()
    return (resp.json() or {}).get("results") or []


def search_web(query: str, max_results: int | None = None,
               search_depth: str | None = None) -> list[dict]:
    """Search the web; returns normalized web chunks (possibly []).

    Never raises: on any failure an empty list is returned and the
    caller continues with local-only results.
    """
    if not web_search_available():
        return []
    query = (query or "").strip()
    if not query:
        return []
    max_results = max_results or settings.TAVILY_MAX_RESULTS
    search_depth = (search_depth or settings.TAVILY_SEARCH_DEPTH or "basic").lower()
    if search_depth not in _VALID_DEPTHS:
        search_depth = "basic"
    try:
        try:
            results = _search_via_sdk(query, max_results, search_depth)
            backend = "tavily-python"
        except ImportError:
            results = _search_via_http(query, max_results, search_depth)
            backend = "httpx"
    except Exception as exc:  # noqa: BLE001 - web search must never break chat
        logger.warning("Tavily web search failed (%s); continuing without "
                       "web results.", type(exc).__name__)
        return []
    docs = _normalize_results(results)
    logger.info("Tavily web search (%s): %d results for %r",
                backend, len(docs), query[:60])
    return docs


def result_domains(docs: list[dict]) -> list[str]:
    """Sorted unique domains of web chunks, for the run trace."""
    return sorted({urlparse(d.get("source", "")).netloc
                   for d in docs or [] if d.get("source")})

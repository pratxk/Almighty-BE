"""Honest, simple evaluation helpers (used by app/scripts/evaluate.py).

No fake precision: retrieval hit rate is the fraction of expected
keywords found in the retrieved sources; faithfulness is the fraction
of answer sentences sharing keywords with the sources. Both are
keyword-overlap proxies, labelled as such in their output.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "it", "its", "this", "that", "with", "for", "on", "as", "by", "at", "from",
    "what", "which", "how", "why", "when", "who", "does", "do", "did", "can",
}


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def evaluate_answer(question: str, expected: str, answer: str,
                    sources: list[dict] | list[str] | None = None) -> dict:
    """Score one answer against an expected answer / retrieved sources.

    ``expected`` is the reference answer (or a comma/space-separated list
    of expected keywords). ``sources`` is the retrieved-doc list
    (dicts with ``text``/``snippet``) or plain strings.
    """
    expected_keywords = _keywords(expected)
    source_text = " ".join(
        (s.get("text") or s.get("snippet") or "") if isinstance(s, dict) else str(s)
        for s in (sources or []))
    source_words = _keywords(source_text)

    if expected_keywords:
        hits = expected_keywords & source_words
        retrieval_hit_rate = len(hits) / len(expected_keywords)
    else:
        hits, retrieval_hit_rate = set(), 0.0

    sentences = _split_sentences(answer)
    supported = 0
    for sentence in sentences:
        words = _keywords(sentence)
        if not words:
            supported += 1  # empty / punctuation-only sentence: not a claim
        elif words & source_words:
            supported += 1
    faithfulness = (supported / len(sentences)) if sentences else 0.0

    return {
        "question": question,
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "retrieval_hits": sorted(hits),
        "retrieval_missed": sorted(expected_keywords - source_words),
        "faithfulness": round(faithfulness, 3),
        "sentences_total": len(sentences),
        "sentences_supported": supported,
        "note": "Keyword-overlap proxies, not human judgement.",
    }

#!/usr/bin/env python3
"""Evaluate the Adaptive RAG API against data/dataset.json.

Usage:
    python backend/scripts/evaluate.py [--api-url http://localhost:8000]
                                       [--dataset data/dataset.json]

For each question it POSTs /api/chat and measures:
  - retrieval hit rate: fraction of expected_keywords found in the
    returned sources' snippets/titles (case-insensitive substring match)
  - faithfulness: fraction of answer sentences sharing at least one
    keyword with the sources (a sentence counts if any expected keyword
    -- or any word from the source text -- appears in it; here we use
    expected keywords as the shared vocabulary, per the spec)
  - answered: whether an answer was returned at all

Prints a per-question table and aggregate averages, and writes
backend/eval-results.json. Uses only the standard library (urllib).

If the API is unreachable, exits with a clear error and no fake scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Repo root is two levels up from this script: backend/scripts/evaluate.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "eval-results.json"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? followed by whitespace/end."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "it", "its", "this", "that", "with", "for", "on", "as", "by", "at", "from",
    "what", "which", "how", "why", "when", "who", "does", "do", "did", "can",
    "is", "tell", "me", "about", "please", "explain", "based", "using", "used",
    "use", "one", "also", "more", "such", "than", "then", "into", "like",
    "answer", "question", "context", "provided", "following", "below",
}


def content_words(text: str) -> list[str]:
    """Lowercased content words (len>=4, not stopwords) of a sentence."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 4 and w not in _STOPWORDS]


def source_text(sources: list) -> str:
    """Concatenate the searchable text of all returned sources."""
    bits: list[str] = []
    for src in sources:
        if not isinstance(src, dict):
            bits.append(str(src))
            continue
        for key in ("snippet", "text", "content", "title", "source", "doc", "filename"):
            if src.get(key):
                bits.append(str(src[key]))
    return " ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Adaptive RAG API.")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the backend API (default: http://localhost:8000)")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="Path to dataset.json (default: data/dataset.json)")
    parser.add_argument("--delay-seconds", type=float, default=0.0,
                        help="Pause between questions (default 0). The Gemini "
                             "free tier allows ~10 generate calls/min, so use "
                             "--delay-seconds 7 for a full 24-question run to "
                             "stay under the limit instead of hitting 429s.")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    dataset_path = Path(args.dataset)

    if not dataset_path.is_file():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in dataset {dataset_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(dataset, list) or not dataset:
        print(f"ERROR: dataset must be a non-empty JSON list: {dataset_path}", file=sys.stderr)
        return 2

    # --- Connectivity check: fail fast with a clear error, never fake scores.
    # NOTE: use GET /health, not a /api/chat call — a chat call runs the
    # full LLM pipeline and can legitimately take >10s on a cold start.
    try:
        req = urllib.request.Request(f"{api_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError:
        # An HTTP response (even 4xx/5xx) proves the server is reachable.
        pass
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"ERROR: API unreachable at {api_url}: {exc}", file=sys.stderr)
        print("Start the backend first (see README: docker compose up, or run the "
              "FastAPI app locally), then re-run this script.", file=sys.stderr)
        return 1

    results: list[dict] = []

    for i, item in enumerate(dataset):
        if i and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
        qid = item.get("id", "?")
        question = item.get("question", "")
        keywords = [k.lower() for k in item.get("expected_keywords", [])]

        payload = json.dumps({"question": question}).encode()
        req = urllib.request.Request(f"{api_url}/api/chat", method="POST", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            print(f"[{qid}] HTTP {exc.code} from /api/chat: {detail}", file=sys.stderr)
            body = {}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"ERROR: API became unreachable at {api_url} during question {qid}: {exc}",
                  file=sys.stderr)
            return 1

        # Accept common response shapes: {"answer": ..., "sources": [...]}
        # or {"response": ...} / {"result": ...}; sources may be under
        # "sources", "retrieved", or "context".
        answer = ""
        for key in ("answer", "response", "result", "output"):
            if isinstance(body.get(key), str) and body[key].strip():
                answer = body[key]
                break
        sources = []
        for key in ("sources", "retrieved", "context", "documents"):
            if isinstance(body.get(key), list):
                sources = body[key]
                break

        answered = bool(answer.strip())
        src_text = source_text(sources).lower()

        # (a) Retrieval hit rate: expected keywords found in sources.
        if keywords:
            hits = sum(1 for kw in keywords if kw in src_text)
            hit_rate = hits / len(keywords)
        else:
            hits, hit_rate = 0, 0.0

        # (b) Faithfulness (keyword-based, kept for comparability):
        # answer sentences sharing >=1 keyword with sources.
        # A sentence counts if it contains a keyword that also appears in
        # the retrieved sources -- i.e. the claim is grounded in retrieval.
        sentences = split_sentences(answer) if answered else []
        grounded_keywords = {kw for kw in keywords if kw in src_text}
        if sentences:
            supported = sum(1 for s in sentences
                            if any(kw in s.lower() for kw in grounded_keywords))
            faithfulness = supported / len(sentences)
        else:
            supported, faithfulness = 0, 0.0

        # (c) Faithfulness (word-overlap, fairer): a sentence counts as
        # grounded if >=50% of its content words appear in the source
        # text, or it contains an expected keyword. The keyword-only
        # metric undercounts real grounding: LLM answers paraphrase
        # ("routing the query" vs keyword "query routing"), and list
        # items often carry no keyword at all while still being verbatim
        # from a source. Both numbers are reported; neither is hidden.
        if sentences:
            src_words = set(re.findall(r"[a-z0-9]+", src_text))
            supported_w = 0
            for s in sentences:
                cw = content_words(s)
                if not cw:
                    supported_w += 1
                    continue
                shared = sum(1 for w in cw if w in src_words)
                if (shared / len(cw) >= 0.5
                        or any(kw in s.lower() for kw in grounded_keywords)):
                    supported_w += 1
            faithfulness_words = supported_w / len(sentences)
        else:
            supported_w, faithfulness_words = 0, 0.0

        results.append({
            "id": qid,
            "question": question,
            "source_doc": item.get("source_doc"),
            "answered": answered,
            "retrieval_hit_rate": round(hit_rate, 3),
            "keywords_hit": f"{hits}/{len(keywords)}",
            "faithfulness": round(faithfulness, 3),
            "faithfulness_words": round(faithfulness_words, 3),
            "sentences_supported": f"{supported}/{len(sentences)}",
            "sentences_supported_words": f"{supported_w}/{len(sentences)}",
            "num_sources": len(sources),
            "answer": answer,
        })

    # --- Print per-question table.
    print(f"\n{'ID':<14} {'Answered':<9} {'Hit rate':<9} {'Faithful':<9} {'FaithW':<8} {'Sources':<8} Question")
    print("-" * 118)
    for r in results:
        print(f"{r['id']:<14} {str(r['answered']):<9} "
              f"{r['retrieval_hit_rate']:<9.3f} {r['faithfulness']:<9.3f} "
              f"{r['faithfulness_words']:<8.3f} {r['num_sources']:<8} {r['question'][:60]}")

    n = len(results)
    avg_hit = sum(r["retrieval_hit_rate"] for r in results) / n
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_faith_w = sum(r["faithfulness_words"] for r in results) / n
    answered_rate = sum(1 for r in results if r["answered"]) / n
    print("-" * 118)
    print(f"Questions: {n}")
    print(f"Answered rate:        {answered_rate:.3f} ({sum(1 for r in results if r['answered'])}/{n})")
    print(f"Avg retrieval hit rate: {avg_hit:.3f}  (fraction of expected keywords found in returned sources)")
    print(f"Avg faithfulness:       {avg_faith:.3f}  (keyword-based: fraction of answer sentences sharing an expected keyword with sources)")
    print(f"Avg faithfulness(words): {avg_faith_w:.3f}  (word-overlap: fraction of answer sentences with >=50% of content words in sources)")

    # --- Write results file.
    output = {
        "api_url": api_url,
        "dataset": str(dataset_path),
        "num_questions": n,
        "aggregate": {
            "answered_rate": round(answered_rate, 3),
            "avg_retrieval_hit_rate": round(avg_hit, 3),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_faithfulness_words": round(avg_faith_w, 3),
        },
        "results": results,
    }
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the Adaptive RAG API with RAGAS (the standard RAG eval library).

Replaces/augments the hand-rolled keyword metrics in evaluate.py with
grounded, judge-based metrics: faithfulness, answer_relevancy,
context_precision, context_recall.

Phases (run together by default, or separately):
  --ingest  ingest the dataset's docs into the running API first
  --collect ask every question via POST /api/chat, save raw samples
  --score   score the raw samples with RAGAS (Gemini judge + embeddings)

Usage:
    python backend/scripts/evaluate_ragas.py --dataset data/dataset.json
        [--api-url http://localhost:8001] [--delay-seconds 7]
        [--ingest] [--collect-only | --score-only --raw-file ...]

Outputs:
  backend/eval-results-ragas/<name>-raw-<ts>.json   (answers + contexts)
  backend/eval-results-ragas/<name>-ragas-<ts>.json (per-sample + aggregates)
  backend/eval-history.jsonl                        (one row per run)

Notes:
- The Gemini key is read via app.config.settings (same source as the app);
  it is never printed or logged.
- If LANGSMITH_API_KEY is set, the dataset + examples are also uploaded
  to LangSmith; otherwise that step is skipped cleanly.
- Run against a server with a FRESH corpus for the dataset (fresh DB +
  fresh BM25 + fresh Qdrant collection), or scores will mix corpora.
- RAGAS 0.4.x needs the venv shims from apply_venv_shims.py
  (langchain-community removed the VertexAI modules ragas imports).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "eval-results-ragas"
HISTORY_FILE = BACKEND_DIR / "eval-history.jsonl"
DOCS_DIR = REPO_ROOT / "data" / "sample-docs"

METRIC_NAMES = ["faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# HTTP helpers (stdlib only, like evaluate.py)
# --------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_with_retry(url: str, payload: dict, timeout: int = 120,
                     tries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return _post_json(url, payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last = RuntimeError(f"HTTP {exc.code}: {body}")
            _log(f"  retry {attempt + 1}/{tries} after HTTP {exc.code}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            _log(f"  retry {attempt + 1}/{tries} after {exc}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {tries} tries: {last}")


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------
# Old (keyword) metrics — kept so RAGAS can be compared apples-to-apples
# --------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _old_metrics(answer: str, sources: list, keywords: list[str]) -> dict:
    src_text = " ".join(
        str(s.get(k, "")) for s in sources if isinstance(s, dict)
        for k in ("snippet", "text", "title", "source")).lower()
    kws = [k.lower() for k in keywords]
    hits = sum(1 for kw in kws if kw in src_text)
    hit_rate = hits / len(kws) if kws else 0.0
    sentences = _split_sentences(answer) if answer.strip() else []
    grounded = {kw for kw in kws if kw in src_text}
    if sentences:
        supported = sum(1 for s in sentences
                        if any(kw in s.lower() for kw in grounded))
        faith_kw = supported / len(sentences)
        src_words = set(re.findall(r"[a-z0-9]+", src_text))
        supported_w = 0
        for s in sentences:
            cw = [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4]
            if not cw:
                supported_w += 1
                continue
            shared = sum(1 for w in cw if w in src_words)
            if shared / len(cw) >= 0.5 or any(kw in s.lower() for kw in grounded):
                supported_w += 1
        faith_words = supported_w / len(sentences)
    else:
        faith_kw = faith_words = 0.0
    return {"retrieval_hit_rate": round(hit_rate, 3),
            "faithfulness_keyword": round(faith_kw, 3),
            "faithfulness_words": round(faith_words, 3),
            "num_sources": len(sources)}


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def _doc_files_for(dataset_path: Path, dataset: list) -> list[Path]:
    name = dataset_path.stem
    if "hotpotqa" in name:
        return sorted((DOCS_DIR / "hotpotqa").glob("*.md"))
    if "squad" in name:
        return sorted((DOCS_DIR / "squad").glob("*.md"))
    files: list[Path] = []
    for item in dataset:
        for key in ("source_doc",):
            f = item.get(key)
            if f:
                p = DOCS_DIR / f
                if p not in files:
                    files.append(p)
        for f in item.get("source_docs", []) or []:
            p = DOCS_DIR / f
            if p not in files:
                files.append(p)
    return files


def _ingest_file(api_url: str, path: Path) -> dict:
    boundary = "----ragas-eval-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: text/markdown\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{api_url}/api/ingest", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def do_ingest(api_url: str, dataset_path: Path, dataset: list,
              delay: float) -> None:
    files = _doc_files_for(dataset_path, dataset)
    _log(f"Ingesting {len(files)} doc files into {api_url} ...")
    for i, path in enumerate(files):
        if i and delay:
            time.sleep(delay)
        if not path.exists():
            _log(f"  SKIP missing {path}")
            continue
        try:
            res = _ingest_file(api_url, path)
            stats = res.get("stats", {})
            _log(f"  [{i + 1}/{len(files)}] {path.name}: "
                 f"{stats.get('chunks', '?')} chunks")
        except Exception as exc:  # noqa: BLE001 - report and continue
            _log(f"  FAIL {path.name}: {exc}")
    try:
        docs = _get_json(f"{api_url}/api/documents")
        _log(f"Corpus now: {docs.get('corpus', {})}")
    except Exception as exc:  # noqa: BLE001
        _log(f"  (could not read /api/documents: {exc})")


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

def do_collect(api_url: str, dataset: list, delay: float,
               skip_no_doc: bool) -> list[dict]:
    samples: list[dict] = []
    for i, item in enumerate(dataset):
        if skip_no_doc and not item.get("source_doc"):
            _log(f"[{item.get('id', '?')}] SKIP (unanswerable / no source doc)")
            continue
        if i and delay:
            time.sleep(delay)
        qid = item.get("id", "?")
        question = item.get("question", "")
        t0 = time.time()
        try:
            body = _post_with_retry(f"{api_url}/api/chat",
                                    {"question": question})
        except Exception as exc:  # noqa: BLE001
            _log(f"[{qid}] ERROR: {exc}")
            samples.append({"id": qid, "question": question,
                            "answer": "", "contexts": [],
                            "ground_truth": item.get("expected_answer", ""),
                            "error": str(exc)})
            continue
        answer = body.get("answer", "") or ""
        sources = body.get("sources", []) or []
        contexts = [" ".join(str(s.get("snippet", "")).split())
                    for s in sources if isinstance(s, dict) and s.get("snippet")]
        old = _old_metrics(answer, sources, item.get("expected_keywords", []))
        samples.append({
            "id": qid, "question": question, "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("expected_answer", ""),
            "expected_keywords": item.get("expected_keywords", []),
            "route": body.get("route"), "confidence": body.get("confidence"),
            "latency_s": round(time.time() - t0, 1), **old,
        })
        _log(f"[{qid}] answered={bool(answer.strip())} "
             f"hit={old['retrieval_hit_rate']} faith_kw={old['faithfulness_keyword']} "
             f"({len(contexts)} ctx, {samples[-1]['latency_s']}s)")
    return samples


# --------------------------------------------------------------------------
# Score with RAGAS
# --------------------------------------------------------------------------

def _judge_clients(judge_model: str):
    """Judge LLM + embeddings for RAGAS 0.4.x evaluate().

    The legacy metric singletons (ragas.metrics._*) used by evaluate()
    accept the classic LangChain wrappers. (The newer collections
    metrics demand InstructorLLM/modern embeddings and are NOT
    compatible with evaluate() at all in 0.4.3 — see report.)

    Default judge model is gemini-2.0-flash, NOT the app's
    gemini-3.5-flash-lite: answer_relevancy asks the judge for multiple
    candidates and flash-lite rejects that ("Multiple candidates is not
    enabled for this model"). Override with --judge-model.
    """
    from app.config import settings

    key = settings.GEMINI_API_KEY
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (backend/.env); "
                           "RAGAS judging needs it.")
    # Judge embeddings: local MiniLM, NOT Google — the Gemini embedding
    # daily free-tier quota (1000 calls/day) is exhausted until reset, so
    # any Google embedding call 429s. MiniLM is the app's own local
    # provider (settings.EMBEDDING_MODEL) and needs no quota.
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=judge_model, google_api_key=key,
        temperature=0, max_retries=5, timeout=180))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL))
    return llm, embeddings


def do_score(raw_path: Path, out_path: Path, max_workers: int,
             judge_model: str) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    samples = raw["samples"]
    usable = [s for s in samples if s.get("answer", "").strip()
              and s.get("contexts")]
    skipped = len(samples) - len(usable)
    if skipped:
        _log(f"Scoring {len(usable)}/{len(samples)} samples "
             f"({skipped} skipped: empty answer or no contexts)")
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    # Legacy singletons: the only metrics evaluate() accepts in 0.4.3
    # (the newer collections metrics are not Metric instances and are
    # built for the @experiment flow, not evaluate()).
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness

    ragas_samples = [SingleTurnSample(
        user_input=s["question"], response=s["answer"],
        retrieved_contexts=s["contexts"], reference=s["ground_truth"])
        for s in usable]
    llm, embeddings = _judge_clients(judge_model)
    metrics = [faithfulness, answer_relevancy,
               context_precision, context_recall]
    _log(f"RAGAS scoring {len(ragas_samples)} samples "
         f"(workers={max_workers}) — this makes several judge calls per sample ...")
    result = evaluate(
        dataset=EvaluationDataset(samples=ragas_samples),
        metrics=metrics, llm=llm, embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers, max_retries=10,
                             timeout=300),
        show_progress=True)
    df = result.to_pandas()
    per_sample = []
    for (i, s), (_, row) in zip(enumerate(usable), df.iterrows()):
        entry = {"id": s["id"], "question": s["question"]}
        for m in METRIC_NAMES:
            v = row.get(m)
            entry[m] = None if v is None or (isinstance(v, float)
                                             and math.isnan(v)) else round(float(v), 3)
        per_sample.append(entry)
    agg = {}
    for m in METRIC_NAMES:
        vals = [e[m] for e in per_sample if e[m] is not None]
        agg[m] = round(sum(vals) / len(vals), 3) if vals else None
        agg[f"{m}_n"] = len(vals)
    old_agg = {}
    for m in ("retrieval_hit_rate", "faithfulness_keyword", "faithfulness_words"):
        vals = [s[m] for s in usable if s.get(m) is not None]
        old_agg[m] = round(sum(vals) / len(vals), 3) if vals else None
    scored = {"dataset": raw["dataset"], "framework": "ragas",
              "ragas_version": __import__("ragas").__version__,
              "judge_model": judge_model,
              "n_scored": len(usable), "n_skipped": skipped,
              "aggregate": agg, "old_metrics_aggregate": old_agg,
              "per_sample": per_sample}
    out_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    _log(f"RAGAS aggregates: {agg}")
    _log(f"Old-metric aggregates (same samples): {old_agg}")
    return scored


# --------------------------------------------------------------------------
# LangSmith (optional — skipped cleanly without a key)
# --------------------------------------------------------------------------

def maybe_upload_langsmith(dataset_name: str, raw: dict,
                           scored: dict | None) -> str:
    try:
        from app.config import settings
    except Exception:  # noqa: BLE001
        return "skipped (settings unavailable)"
    if not settings.LANGSMITH_API_KEY:
        return "skipped (LANGSMITH_API_KEY not set)"
    try:
        from langsmith import Client

        client = Client(api_key=settings.LANGSMITH_API_KEY)
        name = f"adaptive-rag-evals/{dataset_name}"
        try:
            ds = client.create_dataset(
                dataset_name=name,
                description=f"RAGAS eval set for adaptive-rag ({dataset_name})")
        except Exception:  # noqa: BLE001 - dataset may already exist
            ds = client.read_dataset(dataset_name=name)
        examples = [{
            "inputs": {"question": s["question"]},
            "outputs": {"answer": s.get("answer", ""),
                        "contexts": s.get("contexts", [])},
            "metadata": {"id": s["id"],
                         "ground_truth": s.get("ground_truth", "")},
        } for s in raw["samples"]]
        client.create_examples(dataset_id=ds.id, examples=examples)
        return f"uploaded {len(examples)} examples to LangSmith dataset '{name}'"
    except Exception as exc:  # noqa: BLE001 - never fail the eval on this
        return f"failed (non-fatal): {exc}"


def append_history(row: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="RAGAS evaluation for adaptive RAG.")
    ap.add_argument("--dataset", required=True, help="dataset JSON file")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--delay-seconds", type=float, default=7.0,
                    help="pause between questions / ingests (Gemini free tier)")
    ap.add_argument("--ingest", action="store_true",
                    help="ingest the dataset docs before collecting")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--raw-file", default="",
                    help="raw samples JSON (for --score-only)")
    ap.add_argument("--skip-no-doc", action="store_true",
                    help="skip items with no source_doc (unanswerable)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="RAGAS parallel workers (default 2; free-tier safe)")
    ap.add_argument("--judge-model", default="gemini-2.0-flash",
                    help="judge model for RAGAS (default gemini-2.0-flash; "
                         "the app's flash-lite cannot do answer_relevancy)")
    ap.add_argument("--limit", type=int, default=0, help="max questions (0=all)")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    name = dataset_path.stem.replace("dataset-", "")
    ts = _ts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    api_url = args.api_url.rstrip("/")
    if not args.score_only:
        try:
            health = _get_json(f"{api_url}/health", timeout=10)
            _log(f"API {api_url} reachable: store={health.get('store')}, "
                 f"llm={health.get('llm_provider')}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: API unreachable at {api_url}: {exc}", file=sys.stderr)
            return 1

    raw_path = RESULTS_DIR / f"{name}-raw-{ts}.json"
    scored = None

    if not args.score_only:
        if args.ingest:
            do_ingest(api_url, dataset_path, dataset, args.delay_seconds)
        samples = do_collect(api_url, dataset, args.delay_seconds,
                             args.skip_no_doc)
        try:
            from app.config import settings as _s
            emb_provider = _s.effective_embedding_provider
        except Exception:  # noqa: BLE001
            emb_provider = "unknown"
        raw = {"dataset": name, "api_url": api_url,
               "embedding_provider": emb_provider,
               "collected_at": ts, "samples": samples}
        raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        _log(f"Raw samples saved: {raw_path}")
        if args.collect_only:
            return 0
    else:
        if not args.raw_file:
            print("ERROR: --score-only needs --raw-file", file=sys.stderr)
            return 2
        raw_path = Path(args.raw_file)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        name = raw.get("dataset", name)

    scored_path = RESULTS_DIR / f"{name}-ragas-{ts}.json"
    scored = do_score(raw_path, scored_path, args.max_workers,
                      args.judge_model)
    _log(f"RAGAS results saved: {scored_path}")

    ls = maybe_upload_langsmith(name, raw, scored)
    _log(f"LangSmith: {ls}")

    append_history({
        "date": ts, "dataset": name, "framework": "ragas",
        "ragas_version": scored["ragas_version"],
        "n_scored": scored["n_scored"],
        "metrics": scored["aggregate"],
        "old_metrics": scored["old_metrics_aggregate"],
        "langsmith": ls,
    })
    _log(f"Appended to {HISTORY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

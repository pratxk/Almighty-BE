#!/usr/bin/env python3
"""Build real benchmark datasets for the adaptive RAG evals.

Downloads small, deterministic samples from public benchmarks on the
Hugging Face Hub (no fabricated questions):

  * HotpotQA (distractor, validation) — multi-hop questions whose answers
    need facts from TWO Wikipedia paragraphs. Corpus = gold paragraphs +
    2 distractor paragraphs per question, so retrieval is non-trivial.
  * SQuAD v1.1 (validation) — factoid questions with short answers and
    one source paragraph each.

Outputs (same JSON list format as data/dataset.json):
  data/dataset-hotpotqa.json, data/dataset-squad.json
Docs (one .md file per paragraph):
  data/sample-docs/hotpotqa/*.md, data/sample-docs/squad/*.md

Usage:  python backend/scripts/build_benchmark_datasets.py [--n 20]
Requires the `datasets` package (in backend/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_HOTPOT = REPO_ROOT / "data" / "sample-docs" / "hotpotqa"
DOCS_SQUAD = REPO_ROOT / "data" / "sample-docs" / "squad"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "it", "its", "this", "that", "with", "for", "on", "as", "by", "at", "from",
    "what", "which", "how", "why", "when", "who", "does", "do", "did", "can",
}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "doc"


def _keywords(answer: str, limit: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower())
             if len(w) >= 4 and w not in _STOPWORDS]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def build_hotpotqa(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items: list[dict] = []
    for row in ds:
        if len(items) >= n:
            break
        if row["type"] not in ("bridge", "comparison"):
            continue
        answer = (row.get("answer") or "").strip()
        if not answer or answer.lower() in ("yes", "no"):
            continue
        # Gold paragraphs: the supporting facts' titles.
        # (datasets 5.x gives columnar dicts, not pairs.)
        gold_titles = []
        for title in row["supporting_facts"]["title"]:
            if title not in gold_titles:
                gold_titles.append(title)
        if len(gold_titles) < 2:
            continue
        # Paragraph lookup: title -> joined sentences.
        paras = {t: " ".join(s) for t, s in zip(
            row["context"]["title"], row["context"]["sentences"])}
        if any(t not in paras for t in gold_titles):
            continue
        # Corpus for this question: gold paragraphs + first 2 distractors.
        distractors = [t for t in paras if t not in gold_titles][:2]
        corpus_titles = gold_titles + distractors
        files = []
        for title in corpus_titles:
            import hashlib

            digest = hashlib.sha1(paras[title].encode("utf-8")).hexdigest()[:8]
            fname = f"{_slug(title)}-{digest}.md"
            path = DOCS_HOTPOT / fname
            if not path.exists():
                path.write_text(f"# {title}\n\n{paras[title]}\n", encoding="utf-8")
            files.append(fname)
        gold_files = [f for t, f in zip(corpus_titles, files) if t in gold_titles]
        items.append({
            "id": f"hotpot-{len(items) + 1:02d}",
            "question": row["question"],
            "expected_answer": answer,
            "expected_keywords": _keywords(answer),
            "source_doc": gold_files[0],
            "source_docs": gold_files,
            "benchmark": "hotpotqa",
            "qtype": row["type"],
        })
    return items


def build_squad(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")
    items: list[dict] = []
    for row in ds:
        if len(items) >= n:
            break
        answers = row.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not texts or not texts[0].strip():
            continue
        title = row.get("title", "squad")
        fname = f"{_slug(title)}-{len(items) + 1:02d}.md"
        path = DOCS_SQUAD / fname
        if not path.exists():
            path.write_text(f"# {title}\n\n{row['context']}\n", encoding="utf-8")
        items.append({
            "id": f"squad-{len(items) + 1:02d}",
            "question": row["question"],
            "expected_answer": texts[0].strip(),
            "expected_keywords": _keywords(texts[0]),
            "source_doc": fname,
            "benchmark": "squad",
        })
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="questions per dataset")
    args = ap.parse_args()
    DOCS_HOTPOT.mkdir(parents=True, exist_ok=True)
    DOCS_SQUAD.mkdir(parents=True, exist_ok=True)
    hotpot = build_hotpotqa(args.n)
    squad = build_squad(args.n)
    (REPO_ROOT / "data" / "dataset-hotpotqa.json").write_text(
        json.dumps(hotpot, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPO_ROOT / "data" / "dataset-squad.json").write_text(
        json.dumps(squad, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"hotpotqa: {len(hotpot)} questions, "
          f"{len(list(DOCS_HOTPOT.glob('*.md')))} doc files")
    print(f"squad: {len(squad)} questions, "
          f"{len(list(DOCS_SQUAD.glob('*.md')))} doc files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Benchmark datasets

Real, public benchmarks — no fabricated questions.

## `dataset-hotpotqa.json` — multi-hop QA (20 questions)

- Source: **HotpotQA**, distractor setting, validation split
  (`hotpotqa/hotpot_qa`, Hugging Face Hub).
- Each question needs facts from **two** Wikipedia paragraphs
  (`supporting_facts`); the corpus holds those gold paragraphs plus
  2 distractor paragraphs per question, so retrieval must find the
  right 2 among ~80 paragraphs.
- Docs: `data/sample-docs/hotpotqa/*.md` (one file per paragraph,
  filename `<slug>-<sha1:8>.md`).
- Built by `backend/scripts/build_benchmark_datasets.py`.

## `dataset-squad.json` — factoid QA (20 questions)

- Source: **SQuAD v1.1**, validation split (`rajpurkar/squad`,
  Hugging Face Hub).
- Each question has a short answer span and one source paragraph.
- Docs: `data/sample-docs/squad/*.md` (one file per paragraph).

## Older datasets (project-built, not benchmarks)

- `dataset.json` — 24 questions on RAG / AI agents / Python async
  (docs: `rag-explained.md`, `ai-agents.md`, `python-async.md`).
- `dataset-a.json` — 18 questions on system design (caching, DB
  indexing, API design).
- `dataset-b.json` — 18 questions on personal finance, including 5
  cross-document and 3 unanswerable questions (answerable-only = 15).

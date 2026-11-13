# Adaptive RAG

**An adaptive retrieval-augmented generation system that routes, grades, rewrites, and self-corrects — so answers are grounded in your documents, not hallucinated.**

Adaptive RAG is a full-stack question-answering system over your own documents. Ask a question in the Next.js chat UI and a LangGraph workflow decides how to handle it, retrieves evidence with hybrid search, grades that evidence, rewrites the query when retrieval is weak, and checks the final answer against its sources before showing it to you — with every step visible in the UI.

## Architecture

```
User
  |
  v
Next.js frontend (port 3000) — chat UI, React Flow agent visualisation,
|                              ingestion / token transparency panel
  v  REST: POST /api/chat, GET /api/runs/{id}
FastAPI backend (port 8000)
  |
  v
LangGraph workflow — route -> retrieve -> grade -> rewrite (loop) -> generate
  |                    |                -> self-correct / grounding check
  |                    v
  |              Qdrant hybrid search (vector + BM25) + reranking
  |              (Qdrant on ports 6333/6334, in-memory fallback if absent)
  v
LLM — Google Gemini (default, free tier) / Ollama fallback
  |
  +-- SQLite / Postgres (metadata: runs, chunks, grades)
  +-- LangSmith + OpenTelemetry / Jaeger (tracing, ports 16686/4318)
```

## How Adaptive RAG Works

Adaptive RAG is basic RAG with a feedback loop at every stage. The pipeline is a LangGraph state graph, so each step can inspect the shared state and change what happens next.

- **Query routing.** Before retrieval, a router classifies the question: a factual question goes to the vector store, a question needing exact terms goes to hybrid search, a complex question is decomposed into sub-questions retrieved separately, and an out-of-scope question can be answered directly or refused instead of wasting a retrieval round.
- **Hybrid search.** Every retrieval runs both vector search (meaning, via embeddings) and BM25 keyword search (exact terms), merging the ranked lists with reciprocal rank fusion. A reranker then reorders the top candidates and keeps the best few for the prompt.
- **Relevance grading.** Each retrieved chunk is graded relevant or irrelevant. Irrelevant chunks are discarded before generation so they cannot distract the model or waste context. The UI shows how many chunks were retrieved, kept, and discarded.
- **Query rewriting.** If too few chunks survive grading, the query is rewritten — expanding abbreviations, adding synonyms, making a vague question specific — and retrieval runs again. The loop repeats up to a fixed limit (typically 2–3 attempts), then the pipeline proceeds with what it has or admits it cannot answer.
- **Self-correction loop.** After drafting an answer, a grounding check compares every claim against the retrieved sources (sentence-level: does this sentence share evidence with a source?). A failed check triggers a retry with stricter instructions, more context, or an answer flagged as partly unsupported.
- **Grounding check.** The final answer is only returned with its sources attached. Every answer can be traced to the chunks it came from; every discarded chunk and every rewrite is logged in the run trace.
- **Dynamic top-k.** The number of chunks retrieved adapts to the question: simple factual questions use a small top-k, complex or multi-part questions fetch more candidates before grading and reranking narrow them down.
- **Multi-turn conversation (contextualize-then-retrieve).** The chat sends the last few turns with each question (`history` in the `/api/chat` body). A follow-up like "tell me more about the second one" is first rewritten into a standalone question against that history, so retrieval runs on the resolved question instead of pronouns; the turns also ride along in the generation prompt, clearly separated from the retrieved context. Standalone questions skip the rewrite entirely (no extra LLM call, identical behaviour).
- **Prompt token budget.** The assembled generation prompt is capped by `MAX_CONTEXT_TOKENS` (default 8000, chars/4 estimate): oldest history turns drop first, then the lowest-graded chunks, always keeping at least one chunk. What was dropped is recorded in the run trace (`trace.context_budget`) and shown in the transparency panel — truncation is visible, never silent. Generation itself is capped by `MAX_OUTPUT_TOKENS` (default 1024).

## Agent Visualisation & Transparency (Frontend)

The frontend is not a black box — it shows the pipeline working, live, for every question.

- **React Flow agent visualisation.** Each LangGraph node (route, retrieve, grade, rewrite, generate, self-correct) is rendered as a node in a React Flow graph. Per-node status is shown as the run progresses — pending, running, completed, or failed — with the node's inputs/outputs (for example, which route was chosen, how many chunks each retrieval returned) inspectable on click.
- **Ingestion / token transparency panel.** A side panel shows, for the current run: chunks retrieved / kept / discarded, the query rewrites that happened, and token counts (prompt tokens, completion tokens, and total, per model call where the provider reports them). It also shows which documents were ingested, their chunk counts, and the embedding model used.
- **Run trace.** Every run is stored with an ID. Fetch the full trace any time with `GET /api/runs/{id}` — it contains the node-by-node trace, routing decision, grades, rewrites, retrieved chunks, token usage, and the grounding-check result. The same traces are also exported to LangSmith (if configured) and to Jaeger via OpenTelemetry.

## Gemini as Default (Free Tier)

Google Gemini (Google AI Studio free tier) is the **default LLM and embeddings provider**. It is configured only via `GEMINI_API_KEY` in `backend/.env` — never hardcoded.

### How to get a free key

1. Go to [Google AI Studio](https://aistudio.google.com) and sign in with a Google account.
2. Click **"Get API key"** and create a new API key.
3. Copy it into `backend/.env` as:
   ```bash
   GEMINI_API_KEY=your-key-here
   ```
4. Restart the backend (`docker compose restart backend`, or restart the local dev server).

Defaults: `GEMINI_MODEL=gemini-3.5-flash-lite`, `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`, `EMBEDDING_PROVIDER=gemini`. You can change the model in `backend/.env` without touching code.

**Free-tier limits note:** the Google AI Studio free tier has rate limits and daily quotas (see AI Studio for current limits). If you hit a rate limit, the backend surfaces the provider error rather than silently degrading; wait and retry, or switch to the Ollama fallback below. The key is personal — do not share it and never commit `backend/.env` (it is gitignored).

**No key? Degraded / demo mode.** The system runs without a key: retrieval (Qdrant + BM25) and relevance grading work, and generation falls back to extractive passages — the most relevant retrieved chunks, presented as quoted passages — and the UI says so explicitly. You get real retrieval and real sources, just no LLM-generated prose until a key is added.

**Ollama fallback.** Ollama is the documented fallback: run `ollama pull llama3.1`, set `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=llama3.1` in `backend/.env`, and the backend uses the local model if Gemini is unavailable.

## Free-Tier / Zero-Dollar Stack

| Component | Default (free) | Optional free tier |
| --- | --- | --- |
| LLM + embeddings | Google Gemini, AI Studio free tier (`GEMINI_API_KEY`) | Ollama (local, no key) |
| Vector search | Qdrant via Docker (local) | Qdrant Cloud (free 1 GB cluster) |
| Keyword search | BM25, built in (in-memory index) | — |
| Metadata DB | SQLite (`adaptive_rag.db`) | Neon Postgres (free tier) |
| Tracing | Jaeger via Docker (local) | LangSmith (free tier) |
| Frontend / backend hosting | Local / Docker | — |

The system runs locally with **zero keys** in degraded/demo mode (in-memory fallback + extractive answers, see above). LangSmith, Neon, and Qdrant Cloud are optional free tiers — none of them is required.

## Quick Start

### Docker (recommended)

```bash
cp .env.example backend/.env
# Edit backend/.env and set GEMINI_API_KEY (optional — see "No key?" above)
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard
- Jaeger UI: http://localhost:16686

### Local development

```bash
# Backend
cp .env.example backend/.env   # set GEMINI_API_KEY in backend/.env
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (in a second terminal)
cd frontend
npm install
npm run dev   # http://localhost:3000
```

Qdrant and Jaeger are optional locally: without Qdrant the backend uses its in-memory vector fallback, and without Jaeger tracing is simply not exported. Start them with `docker compose up qdrant jaeger` if you want them.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Google AI Studio key (default LLM + embeddings). Optional; no key = demo mode. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini chat model. |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model. |
| `EMBEDDING_PROVIDER` | `gemini` | Embedding provider (`gemini` or `ollama`). |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL. |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key (only for Qdrant Cloud). |
| `DATABASE_URL` | `sqlite+aiosqlite:///./adaptive_rag.db` | Metadata DB (SQLite default; Neon Postgres URL optional). |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama-compatible base URL (fallback LLM). |
| `LLM_MODEL` | `llama3.1` | Fallback LLM model name. |
| `OPENAI_API_KEY` | — | Optional; only if you switch the provider to OpenAI. |
| `LANGSMITH_API_KEY` | — | Optional LangSmith tracing key. |
| `LANGSMITH_PROJECT` | `adaptive-rag` | LangSmith project name. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | Jaeger OTLP endpoint (leave unset to disable). |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend. |

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | Ask a question. Body: `{"question": "..."}`. Optional `"history": [{"role": "user"\|"assistant", "content": "..."}]` (last turns, for follow-ups). Returns `{"answer": "...", "sources": [...], "run_id": "..."}`. |
| `GET` | `/api/runs/{id}` | Full run trace: node statuses, routing decision, grades, rewrites, chunks kept/discarded, token counts, grounding check. |
| `POST` | `/api/ingest` | Ingest documents (upload or point at a directory). Returns chunk counts per document. |
| `GET` | `/api/documents` | List ingested documents and their chunk counts. |
| `GET` | `/api/health` | Health check (`{"status": "ok", ...}`). |

## How to Ingest Docs

Drop Markdown (or text) files into `data/sample-docs/` — four samples are included (AI agents, RAG, Python async, adaptive RAG) — then ingest them:

```bash
# Via the API (backend running):
curl -X POST http://localhost:8000/api/ingest -F "files=@data/sample-docs/rag-explained.md"
# Or ingest the whole sample directory:
curl -X POST http://localhost:8000/api/ingest -H "Content-Type: application/json" \
  -d '{"directory": "data/sample-docs"}'
```

Ingestion chunks each document (500–1,000 characters with overlap), embeds the chunks with the configured embedding model (`gemini-embedding-001` by default), stores them in Qdrant, and builds the BM25 keyword index (`data/bm25_index.json`, gitignored). Re-ingesting a document replaces its old chunks. The transparency panel shows ingestion status and chunk counts per document.

## How to Run the Evaluation

The evaluation script (`backend/scripts/evaluate.py`, stdlib only) asks every question in `data/dataset.json` (24 Q&A pairs grounded in the four sample docs) to the running API and scores each answer:

```bash
# Backend must be running first (docker compose up, or local uvicorn)
python backend/scripts/evaluate.py [--api-url http://localhost:8000] [--dataset data/dataset.json]
```

### What the metrics mean

- **Retrieval hit rate** — for each question, the fraction of its `expected_keywords` found in the returned sources' snippets/titles. Measures whether retrieval found the right evidence. Low hit rate = retrieval problem (chunking, embedding, or top-k), not a generation problem.
- **Faithfulness** — the fraction of answer sentences that share at least one keyword with the retrieved sources. Measures whether the answer is grounded in what was retrieved. Low faithfulness with high hit rate = the model is ignoring or inventing beyond its sources.
- **Answered rate** — the fraction of questions that got any answer at all. A pipeline that refuses or errors on many questions shows up here.

The script prints a per-question table plus aggregate averages, and writes the full results (including every answer) to `backend/eval-results.json` (gitignored). If the API is unreachable it exits with a clear error message — it never fabricates scores.

## Project Structure

```
adaptive-rag/
├── README.md
├── LICENSE
├── docker-compose.yml          # qdrant, jaeger, backend, frontend
├── .env.example                # copy to backend/.env; Gemini-first
├── .gitignore
├── data/
│   ├── dataset.json            # 24 Q&A pairs for evaluation
│   └── sample-docs/
│       ├── ai-agents.md        # AI agents, LangGraph, multi-agent systems
│       ├── rag-explained.md    # RAG, embeddings, vector/hybrid search, reranking
│       ├── python-async.md     # Python async/await, FastAPI, concurrency
│       └── adaptive-rag.md     # Routing, grading, rewriting, self-correction
├── backend/
│   ├── scripts/
│   │   └── evaluate.py         # stdlib-only evaluation runner
│   ├── eval-results.json       # generated by evaluate.py (gitignored)
│   └── .env                    # your keys (gitignored; copy from .env.example)
└── frontend/
    └── (Next.js app: chat UI, React Flow agent visualisation,
         ingestion / token transparency panel)
```

## Notes

- LangSmith, Neon, and Qdrant Cloud are optional free tiers. The system runs locally with zero keys: Qdrant falls back to an in-memory store, the database is a local SQLite file, and generation falls back to extractive passages with a clear notice in the UI.
- Never commit `backend/.env` or any real API key. `.env.example` is the only env file in the repo.
- License: MIT — see [LICENSE](LICENSE).

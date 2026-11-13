# Adaptive RAG — Backend

FastAPI + LangChain + LangGraph adaptive RAG system with hybrid retrieval
(Qdrant dense + BM25, reciprocal rank fusion, cosine rerank), relevance
grading, query rewriting with retries, and grounding checks.

**Default LLM/embeddings: Google Gemini (free tier)** via plain `httpx`
REST — no `google-genai` SDK needed. Ollama-compatible endpoints are the
fallback; with no key at all the API still runs in degraded/demo mode
(in-memory store + extractive answers, clearly labelled).

## Run locally

```bash
cd backend
cp .env.example .env        # then set GEMINI_API_KEY (at least) in .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the OpenAPI UI.

No Qdrant needed to start: if `QDRANT_URL` is unreachable the store
falls back to an in-memory store (logged at startup). For persistence,
run Qdrant: `docker run -p 6333:6333 qdrant/qdrant`.

## Environment variables

See `.env.example`. Key ones:

| Var | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Default LLM + embeddings provider (free tier) |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | LLM model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | 768-dim embeddings |
| `EMBEDDING_PROVIDER` | `auto` | `gemini` \| `local` \| `hash` \| `auto` |
| `EMBEDDING_DIM` | provider default | 768 Gemini, 384 local/hash; Qdrant uses this |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama fallback, only if no Gemini key |
| `LLM_MODEL` | `llama3.1` | Ollama model |
| `QDRANT_URL` | `http://localhost:6333` | Falls back to memory store if down |
| `DATABASE_URL` | SQLite `./adaptive_rag.db` | Neon Postgres URL also works |
| `LANGSMITH_API_KEY` / `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Optional tracing |
| `TOP_K` / `MAX_RETRIES` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | 5 / 2 / 1000 / 200 | Retrieval + chunking |

## Endpoints

- `GET /health` — status, active store, providers
- `POST /api/chat` `{question}` — runs the graph; returns answer, confidence, sources, route, retries, `run_id`, and the full `trace`
- `GET /api/runs/{id}` — stored run trace for a run/conversation id
- `POST /api/ingest` — multipart `.pdf` / `.txt` / `.md` upload; returns chunk/token stats
- `POST /api/ingest-url` `{url}` — ingest a web page
- `GET /api/documents` — documents + corpus totals (chars, tokens, chunks)
- `DELETE /api/documents/{id}` — delete a document and its vectors
- `GET /api/conversations` — recent conversations

## Run trace (token transparency)

Every chat response includes a `trace` with: corpus totals (documents
ingested, total chars / approx tokens, total chunks, chunk size/overlap),
chunks retrieved, kept/discarded after grading, the rewritten query,
per-LLM-call token counts (real `usage_metadata`/`usage` when the API
returns it, otherwise chars/4 estimates), and per-node timings.
Traces are kept in memory and their summary persisted on the
conversation row — fetch one later via `GET /api/runs/{id}`.

## Docker

```bash
docker build -t adaptive-rag-backend .
docker run -p 8000:8000 --env-file .env adaptive-rag-backend
```

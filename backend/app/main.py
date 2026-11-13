"""FastAPI app: chat (adaptive RAG), ingestion, documents, runs.

Run ``uvicorn app.main:app --reload`` from ``backend/``.
With no keys and no Qdrant, the app still runs in degraded/demo mode:
in-memory retrieval + extractive answers, clearly labelled as such.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.cache import build_cached_trace, get_response_cache
from app.config import settings
from app.db import (create_conversation, delete_document, get_conversation_by_run,
                    get_corpus_stats, get_document, init_db, list_conversations,
                    list_documents)
from app.graph import run_adaptive_rag, run_adaptive_rag_stream
from app.guardrails import guardrail_status, normalize_roles
from app.ingestion import ingest_file, ingest_url
from app.run_store import get_run_trace, save_run_trace
from app.tracing import setup_tracing
from app.vectorstore import get_vector_store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Adaptive RAG", version="1.0.0",
              description="Adaptive RAG: hybrid retrieval + LangGraph workflow "
                          "(Gemini default, Ollama fallback, extractive demo mode).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: str | None = None
    # Prior turns ({role: "user"|"assistant", content: ...}) for
    # follow-up resolution; optional — standalone questions send none.
    history: list[dict] | None = None
    # Requesting user's role for guardrails (default "employee").
    # Retrieval is filtered to this role and the cache is keyed by it;
    # without it, one role's answer could be served to another.
    role: str = "employee"
    # Force the Tavily web-search fallback for this question even when
    # local documents answer (web chunks are merged with the local ones
    # and graded together). A forced search bypasses the cache lookup
    # so the web call actually happens; the fresh answer is still
    # stored, so a later identical question can hit the cache.
    web_search: bool = False


class IngestUrlRequest(BaseModel):
    url: str
    # Guardrail access roles for the ingested document (e.g. ["exec"]).
    # None means public (["employee"]).
    access_roles: list[str] | None = None


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    setup_tracing()
    store = get_vector_store()
    # A fresh/empty metadata DB must not see stale BM25 chunks left over
    # from a deleted database (the in-memory vector store is rebuilt empty
    # on every boot, but the BM25 file persists on disk).
    stats = await get_corpus_stats()
    if stats.get("documents_ingested", 0) == 0:
        store.bm25.clear()
    cache = get_response_cache()
    logger.info("Adaptive RAG started: store=%s (dim=%d), llm_provider=%s, "
                "embedding_provider=%s, response_cache=%s",
                store.mode, store.dim, settings.effective_llm_provider,
                settings.effective_embedding_provider, cache.mode)


@app.get("/")
def root() -> dict:
    return {"service": "adaptive-rag", "status": "ok",
            "docs": "/docs", "health": "/health",
            "llm_provider": settings.effective_llm_provider,
            "embedding_provider": settings.effective_embedding_provider}


@app.get("/health")
def health() -> dict:
    store = get_vector_store()
    return {"status": "ok", "qdrant": settings.QDRANT_URL, "store": store.mode,
            "collection": settings.QDRANT_COLLECTION, "dim": store.dim,
            "llm_provider": settings.effective_llm_provider,
            "embedding_provider": settings.effective_embedding_provider,
            "cache": get_response_cache().status(),
            "guardrails": guardrail_status()}


def _serve_cached(entry: dict, hit_type: str, question: str, corpus: dict,
                  serve_ms: float, similarity: float | None = None) -> dict:
    """Build the response payload for a cache hit (fresh run id + trace).

    The served trace has a single ``cache_hit`` step and zero LLM calls —
    this serving ran no retrieval and no generation. The original run
    stays inspectable via ``cached_from_run_id``.
    """
    run_id = str(uuid.uuid4())
    trace = build_cached_trace(question, run_id, corpus, entry, hit_type,
                               serve_ms, similarity)
    save_run_trace(trace)
    return {
        "answer": entry["answer"], "confidence": entry["confidence"],
        "sources": entry["sources"], "route": entry["route"],
        "retries": 0, "run_id": run_id, "trace": trace,
        "cached": True, "cache_hit_type": hit_type, "llm_calls": 0,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    corpus = await get_corpus_stats()
    cache = get_response_cache()
    documents = await list_documents()
    # Cache lookup first: an exact or semantic hit answers without
    # retrieval or any LLM call. Lookup is sync (Redis + embeddings),
    # so it runs off the event loop like the graph does. The lookup is
    # role-scoped (guardrails): an exec's cached answer must never
    # serve an employee, and vice versa. A forced web search skips the
    # lookup so the web call actually happens.
    hit, hit_type, qemb, similarity = None, None, None, None
    if not req.web_search:
        lookup_started = time.perf_counter()
        hit, hit_type, qemb, similarity = await asyncio.to_thread(
            cache.lookup, req.question, req.history, documents, req.role)
        serve_ms = round((time.perf_counter() - lookup_started) * 1000, 1)
    if hit is not None and hit_type is not None:
        result = _serve_cached(hit, hit_type, req.question, corpus,
                               serve_ms, similarity)
        await create_conversation(
            question=req.question, answer=result["answer"],
            confidence=result["confidence"], sources=result["sources"],
            run_id=result["run_id"], trace=result["trace"])
        return result
    # The graph is synchronous (httpx + LangGraph); run it off the event loop.
    # A failure anywhere in the run (e.g. upstream AI quota) is a 503 with
    # a plain message, never a bare 500 — the run is logged for diagnosis.
    try:
        result = await asyncio.to_thread(run_adaptive_rag, req.question, corpus,
                                         req.history, req.role, req.web_search)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 503, not a 500
        logger.exception("Chat run failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Almighty is temporarily unavailable (the AI provider is "
                   "rate-limited right now); please try again in a little while.")
    # Best-effort store: a failed cache write must never fail the answer.
    try:
        await asyncio.to_thread(cache.store, result, req.question,
                                req.history, documents, qemb, req.role)
    except Exception as exc:  # noqa: BLE001 - caching must not break chat
        logger.warning("Response cache: store failed (%s)", exc)
    await create_conversation(
        question=req.question, answer=result["answer"],
        confidence=result["confidence"], sources=result["sources"],
        run_id=result["run_id"], trace=result["trace"])
    return {
        "answer": result["answer"], "confidence": result["confidence"],
        "sources": result["sources"], "route": result["route"],
        "retries": result["retries"], "run_id": result["run_id"],
        "trace": result["trace"],
        "cached": False,
        "llm_calls": len((result["trace"] or {}).get("llm_calls", [])),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE twin of POST /api/chat — same result, streamed as it happens.

    Events (``data: {...}\\n\\n``), see ``run_adaptive_rag_stream``:
    ``step`` (node running/done), ``token`` (LLM text delta),
    ``done`` (full result — answer, citations, trace, stats), ``error``.
    A ``: ping`` heartbeat comment is sent after 15s without an event, so
    long thinking chains keep the connection (and any proxy) alive instead
    of timing out. The conversation is persisted once the stream finishes.
    """
    corpus = await get_corpus_stats()
    cache = get_response_cache()
    documents = await list_documents()
    hit, hit_type, qemb, similarity = None, None, None, None
    if not req.web_search:
        lookup_started = time.perf_counter()
        hit, hit_type, qemb, similarity = await asyncio.to_thread(
            cache.lookup, req.question, req.history, documents, req.role)
        serve_ms = round((time.perf_counter() - lookup_started) * 1000, 1)
    if hit is not None and hit_type is not None:
        # Cache hit: same SSE shape as a live run (cache_hit step, token
        # chunks, done) so the frontend path is identical — just instant.
        cached = _serve_cached(hit, hit_type, req.question, corpus,
                               serve_ms, similarity)

        async def cached_events():
            try:
                step = cached["trace"]["steps"][0]
                running = {"type": "step", "node": "cache_hit",
                           "step": "cache_hit", "status": "running"}
                done_step = {"type": "step", "node": "cache_hit",
                             "step": "cache_hit", "status": "done",
                             "detail": step.get("detail", ""),
                             "duration_ms": step.get("duration_ms"),
                             "data": step}
                yield f"data: {json.dumps(running)}\n\n"
                yield f"data: {json.dumps(done_step)}\n\n"
                words = str(cached["answer"]).split(" ")
                for i in range(0, len(words), 8):
                    chunk = " ".join(words[i:i + 8])
                    if i + 8 < len(words):
                        chunk += " "
                    token = {"type": "token", "delta": chunk}
                    yield f"data: {json.dumps(token)}\n\n"
                done_event = {"type": "done", **cached}
                yield f"data: {json.dumps(done_event)}\n\n"
            finally:
                try:
                    await create_conversation(
                        question=req.question, answer=cached["answer"],
                        confidence=cached["confidence"],
                        sources=cached["sources"], run_id=cached["run_id"],
                        trace=cached["trace"])
                except Exception as exc:  # noqa: BLE001 - persistence must not break the stream
                    logger.warning("Could not persist cached conversation: %s", exc)

        return StreamingResponse(
            cached_events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})
    events: queue.Queue = queue.Queue()
    done_payload: dict = {}

    def produce() -> None:
        try:
            run_adaptive_rag_stream(req.question, corpus, events.put,
                                    req.history, req.role, req.web_search)
        except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
            logger.exception("Streaming run failed: %s", exc)
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(None)

    async def generate_events():
        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        asyncio.to_thread(events.get), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if await request.is_disconnected():
                        break
                    continue
                if event is None:
                    break
                if event.get("type") == "done":
                    done_payload.update(event)
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            if done_payload:
                try:
                    await create_conversation(
                        question=req.question, answer=done_payload.get("answer", ""),
                        confidence=float(done_payload.get("confidence") or 0.0),
                        sources=done_payload.get("sources") or [],
                        run_id=done_payload.get("run_id", ""),
                        trace=done_payload.get("trace") or {})
                except Exception as exc:  # noqa: BLE001 - persistence must not break the stream
                    logger.warning("Could not persist streamed conversation: %s", exc)
                # Best-effort cache store so the next identical question
                # (or paraphrase) is served from the cache instead.
                # Role-scoped like the lookup (guardrails).
                try:
                    await asyncio.to_thread(
                        cache.store, dict(done_payload), req.question,
                        req.history, documents, qemb, req.role)
                except Exception as exc:  # noqa: BLE001 - caching must not break the stream
                    logger.warning("Response cache: store failed (%s)", exc)
            # The producer thread finishes on its own once the run ends; do
            # not await it here — a disconnected client must not hang the response.
            if not producer.done():
                logger.info("SSE client left; streaming run continues in the background.")

    return StreamingResponse(
        generate_events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    trace = get_run_trace(run_id)
    if trace is not None:
        return {"run_id": run_id, "trace": trace}
    conv = await get_conversation_by_run(run_id)
    if conv is not None and conv.get("trace"):
        return {"run_id": run_id, "trace": conv["trace"],
                "conversation_id": conv["id"]}
    raise HTTPException(status_code=404, detail=f"No run trace found for id: {run_id}")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...),
                 access_roles: str | None = Form(None)) -> dict:
    """Ingest a file. ``access_roles`` is an optional JSON list or
    comma-separated string (e.g. ["exec"] or "hr,exec") tagging the
    document for guardrails; omitted means public (["employee"])."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".txt", ".md", ".markdown"):
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return await ingest_file(tmp_path, source_name=file.filename,
                                 access_roles=_parse_roles_field(access_roles))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _parse_roles_field(raw: str | None):
    """Parse the ingest access_roles form field: a JSON list
    (["exec"]) or a comma-separated string ("hr,exec"). None/"" means
    public. Invalid JSON falls back to comma-splitting."""
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


@app.post("/api/ingest-url")
async def ingest_url_endpoint(req: IngestUrlRequest) -> dict:
    try:
        return await ingest_url(req.url,
                                access_roles=normalize_roles(req.access_roles)
                                if req.access_roles is not None else None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest URL: {exc}")


@app.get("/api/documents")
async def documents() -> dict:
    docs = await list_documents()
    corpus = await get_corpus_stats()
    return {"documents": docs, "corpus": corpus}


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str) -> dict:
    doc = await get_document(doc_id)
    if not await delete_document(doc_id):
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    get_vector_store().delete_by_document(doc_id)
    # Any cached answer grounded in this document is now stale — drop it
    # eagerly (the lookup-time cited-doc check is the backstop).
    if doc is not None:
        try:
            dropped = await asyncio.to_thread(
                get_response_cache().invalidate_document, doc_id,
                doc.get("title", ""), doc.get("source", ""))
            if dropped:
                logger.info("Response cache: dropped %d entries citing "
                            "deleted document %s", dropped, doc_id)
        except Exception as exc:  # noqa: BLE001 - invalidation must not break delete
            logger.warning("Response cache: invalidation failed (%s)", exc)
    return {"deleted": doc_id}


@app.get("/api/conversations")
async def conversations(limit: int = 50) -> dict:
    return {"conversations": await list_conversations(limit=limit)}

"""Background ingest jobs (Redis Queue).

Why a queue: embedding is slow — one Gemini call per chunk used to
take 60s+ for a 46-page PDF, and the HTTP request died before the
ingest finished. Now the API only saves the upload and enqueues a
job (202 straight away); the worker does the slow work and the
frontend polls the job row for progress.

The worker runs in the SAME container as the web server (see
``start.sh``) — no extra Render service, no extra free-tier hours.
Redis is the broker (already the response cache); the job rows live in
Postgres, so a job survives a container restart.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

QUEUE_NAME = "ingest"
# A big document can take a while to embed; 30 minutes is the ceiling
# before RQ marks the job failed (it can be re-queued by hand).
JOB_TIMEOUT_SECONDS = 1800


def get_redis_connection():
    """Redis connection for RQ, from the same REDIS_URL as the cache."""
    import redis

    from app.config import settings

    return redis.Redis.from_url(settings.REDIS_URL)


def get_queue():
    """The ingest queue. Created fresh each call — cheap, and it keeps
    the web process from holding a stale connection."""
    from rq import Queue

    return Queue(QUEUE_NAME, connection=get_redis_connection())


def enqueue_ingest_job(job_id: str) -> None:
    """Hand a job row to the worker. Raises if Redis is unreachable —
    the caller marks the job failed and answers 503."""
    get_queue().enqueue(
        "app.jobs.process_ingest_job",
        job_id,
        job_timeout=JOB_TIMEOUT_SECONDS,
        result_ttl=86400,
        failure_ttl=86400,
    )


def process_ingest_job(job_id: str) -> dict:
    """RQ worker entry point. Must stay a plain module-level function
    (RQ stores it by dotted path). Runs the async pipeline and returns
    the ingest stats, or raises so RQ records the failure."""
    return asyncio.run(_run_job(job_id))


async def _run_job(job_id: str) -> dict:
    """One job, start to finish: fetch the input, ingest it, mark the
    job done (or failed) and drop the stored upload bytes."""
    from app.db import (clear_job_bytes, get_job_with_bytes, update_job)
    from app.ingestion import (chunk_text, extract_text_from_file,
                               fetch_url_text, ingest_text)

    job, file_data = await get_job_with_bytes(job_id)
    if job is None:
        logger.error("Ingest job %s not found; nothing to do.", job_id)
        return {"job_id": job_id, "status": "failed",
                "error": "Job not found."}
    await update_job(job_id, status="processing", progress=5)

    # Progress the pipeline reports: chunked 20, embedded 70, stored 95.
    stage_progress = {"chunked": 20, "embedded": 70, "stored": 95}

    async def report_progress(stage: str, done: int, total: int) -> None:
        await update_job(
            job_id,
            progress=stage_progress.get(stage, 50),
            total_chunks=total,
            done_chunks=done,
        )

    tmp_path: Path | None = None
    try:
        if job["kind"] == "file":
            if not file_data:
                raise ValueError("Upload bytes are missing for this job.")
            suffix = Path(job.get("file_name") or "upload.txt").suffix.lower()
            if suffix not in (".pdf", ".txt", ".md", ".markdown"):
                raise ValueError(
                    f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")
            with tempfile.NamedTemporaryFile(delete=False,
                                           suffix=suffix) as tmp:
                tmp.write(file_data)
                tmp_path = Path(tmp.name)
            text = extract_text_from_file(tmp_path)
            title = Path(job.get("file_name") or "upload").stem
            source = job.get("file_name") or title
        elif job["kind"] == "url":
            title, text = await fetch_url_text(job["source"])
            source = job["source"]
        else:
            raise ValueError(f"Unknown job kind: {job['kind']}")
        if not (text or "").strip():
            raise ValueError("No text could be extracted to ingest.")
        # Chunk count up front so the UI can show real progress even
        # before the first progress callback fires.
        await update_job(job_id, total_chunks=len(chunk_text(text)))
        access_roles = (job.get("meta") or {}).get("access_roles")
        result = await ingest_text(title=title, text=text, source=source,
                                   access_roles=access_roles,
                                   on_progress=report_progress)
        stats = {"document_id": result["document_id"],
                 "chars": result["chars"],
                 "chunks": result["chunks"],
                 "approx_tokens": result["approx_tokens"],
                 "chunk_size": result["chunk_size"],
                 "chunk_overlap": result["chunk_overlap"]}
        await update_job(job_id, status="done", progress=100,
                         done_chunks=result["chunks"], stats=stats)
        logger.info("Ingest job %s done: %s (%d chunks)",
                    job_id, title, result["chunks"])
        return {"job_id": job_id, "status": "done", **stats}
    except Exception as exc:  # noqa: BLE001 - a failed job must record, not crash
        message = str(exc)[:1000]
        logger.exception("Ingest job %s failed: %s", job_id, message)
        await update_job(job_id, status="failed", error=message)
        raise
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        # The bytes did their job — chunks are in Qdrant now.
        await clear_job_bytes(job_id)

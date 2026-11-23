"""Async SQLAlchemy models + helpers.

Works with SQLite (default, ``sqlite+aiosqlite``) and Postgres/Neon
(``postgresql+asyncpg://...``) via ``DATABASE_URL`` — no dialect-specific
code; JSON columns are stored as TEXT so both backends behave the same.

Document rows carry ingestion transparency stats (chars, approx tokens,
chunk count/size/overlap) so ``GET /api/documents`` can report them and
chat traces can summarise the whole corpus. Conversation rows carry the
run id and the run-trace summary JSON for ``GET /api/runs/{id}``.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy import (LargeBinary, String, Text, Float, Integer, DateTime,
                        select, func, text)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.embeddings import estimate_tokens

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(512), default="")
    source: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Ingestion transparency stats
    chars: Mapped[int] = mapped_column(Integer, default=0)
    approx_tokens: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=200)
    # JSON blob (TEXT on both SQLite and Postgres): guardrail metadata,
    # currently {"access_roles": [...]}. Added after the table existed;
    # init_db() ALTERs it in for pre-existing databases.
    meta: Mapped[str] = mapped_column(Text, default="{}")

    def to_dict(self) -> dict:
        try:
            meta = json.loads(self.meta or "{}")
        except json.JSONDecodeError:
            meta = {}
        roles = meta.get("access_roles") or ["employee"]
        return {
            "id": self.id, "title": self.title, "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "chars": self.chars, "approx_tokens": self.approx_tokens,
            "chunks": self.chunk_count, "chunk_count": self.chunk_count,
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
            "access_roles": roles, "meta": meta,
        }


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(36), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sources: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    trace: Mapped[str] = mapped_column(Text, default="{}")    # JSON run-trace summary
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_dict(self) -> dict:
        try:
            sources = json.loads(self.sources or "[]")
        except json.JSONDecodeError:
            sources = []
        try:
            trace = json.loads(self.trace or "{}")
        except json.JSONDecodeError:
            trace = {}
        return {
            "id": self.id, "run_id": self.run_id or self.id,
            "question": self.question, "answer": self.answer,
            "confidence": self.confidence, "sources": sources, "trace": trace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Statuses an ingest job moves through: queued -> processing -> done|failed.
JOB_STATUSES = ("queued", "processing", "done", "failed")


class IngestJob(Base):
    """One background ingest (file or URL), processed by the RQ worker.

    The uploaded file's bytes live here until the worker finishes, then
    are deleted — chunks live in Qdrant, metadata in Postgres, so the
    bytes are only a temporary handoff. This is also what makes jobs
    survive a server restart: the row (and its bytes) are in Postgres,
    not on the container's wiped filesystem.
    """
    __tablename__ = "ingest_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kind: Mapped[str] = mapped_column(String(16), default="file")  # "file" | "url"
    title: Mapped[str] = mapped_column(String(512), default="")
    source: Mapped[str] = mapped_column(String(1024), default="")
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # Stage progress: 0-100. Chunks are filled in once chunking runs.
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    done_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # JSON blob (TEXT, same convention as the other tables): currently
    # {"access_roles": [...]} — the guardrail roles the finished
    # document is tagged with.
    meta: Mapped[str] = mapped_column(Text, default="{}")
    # JSON blob (TEXT): result stats once done, e.g. {"document_id":
    # ..., "chars": ..., "chunks": ..., "approx_tokens": ...}.
    stats: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    # Raw uploaded bytes (BYTEA on Postgres, BLOB on SQLite). Deleted
    # as soon as the job finishes, success or failure.
    file_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    file_name: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow,
                                                 onupdate=_utcnow)

    def to_dict(self) -> dict:
        try:
            stats = json.loads(self.stats or "{}")
        except json.JSONDecodeError:
            stats = {}
        try:
            meta = json.loads(self.meta or "{}")
        except json.JSONDecodeError:
            meta = {}
        return {
            "job_id": self.id, "id": self.id, "kind": self.kind,
            "title": self.title, "source": self.source,
            "status": self.status, "progress": self.progress,
            "total_chunks": self.total_chunks, "done_chunks": self.done_chunks,
            "stats": stats, "error": self.error or None,
            "meta": meta, "has_file": self.file_data is not None,
            "file_name": self.file_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


_engine = None
_session_factory = None


def _normalize_database_url(url: str) -> str:
    """Accept plain Postgres URLs and rewrite them to the async driver
    scheme SQLAlchemy needs. Pratik pastes the URL as ``postgresql://``
    (Supabase's own form); ``create_async_engine`` refuses that and
    needs ``postgresql+asyncpg://``."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


def _get_engine():
    global _engine, _session_factory
    if _engine is None:
        url = _normalize_database_url(settings.DATABASE_URL)
        connect_args = {}
        if url.startswith("postgresql+asyncpg://"):
            # Hosted Postgres (Supabase/Neon) requires TLS. A local
            # Postgres (localhost, e.g. for dev tests) does not use it.
            host = urlparse(url).hostname or ""
            if host not in ("localhost", "127.0.0.1", "::1"):
                connect_args = {"ssl": True}
        _engine = create_async_engine(url, future=True, connect_args=connect_args)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine


def get_session() -> AsyncSession:
    _get_engine()
    return _session_factory()


async def init_db() -> None:
    _get_engine()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Guardrail metadata column, added after the table existed:
    # create_all will not ALTER a pre-existing table, so add it
    # explicitly. It runs in its OWN transaction: on Postgres a failed
    # statement aborts its whole transaction, so a failing ALTER here
    # must not share one with create_all — a fresh DB already has the
    # column, and the failed ALTER would roll the CREATE TABLEs back
    # with it, leaving no tables at all (and no error). On SQLite a
    # failed statement does not abort the transaction, which is why
    # this only ever bit Postgres.
    try:
        async with _engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN meta TEXT DEFAULT '{}'"))
    except Exception:  # noqa: BLE001 - column already present
        pass
    logger.info("DB initialised (%s)", settings.DATABASE_URL.split("://")[0])


# --- documents -----------------------------------------------------------

async def create_document(title: str, source: str, text: str, chunk_count: int,
                          chunk_size: int, chunk_overlap: int,
                          meta: dict | None = None) -> dict:
    chars = len(text)
    async with get_session() as session:
        doc = Document(title=title, source=source, chars=chars,
                       approx_tokens=estimate_tokens(text), chunk_count=chunk_count,
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                       meta=json.dumps(meta or {}))
        session.add(doc)
        await session.commit()
        return doc.to_dict()


async def list_documents(limit: int = 100) -> list[dict]:
    async with get_session() as session:
        rows = (await session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def get_document(doc_id: str) -> dict | None:
    async with get_session() as session:
        row = await session.get(Document, doc_id)
        return row.to_dict() if row else None


async def delete_document(doc_id: str) -> bool:
    async with get_session() as session:
        row = await session.get(Document, doc_id)
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
        return True


async def get_corpus_stats() -> dict:
    """Corpus totals for run traces: docs, chars, approx tokens, chunks."""
    async with get_session() as session:
        row = (await session.execute(select(
            func.count(Document.id),
            func.coalesce(func.sum(Document.chars), 0),
            func.coalesce(func.sum(Document.approx_tokens), 0),
            func.coalesce(func.sum(Document.chunk_count), 0),
        ))).one()
        chunk_row = (await session.execute(select(
            Document.chunk_size, Document.chunk_overlap,
        ).order_by(Document.created_at.desc()).limit(1))).first()
    return {
        "documents_ingested": int(row[0]),
        "total_chars": int(row[1]),
        "total_tokens_est": int(row[2]),
        "total_chunks": int(row[3]),
        "chunk_size": int(chunk_row[0]) if chunk_row else settings.CHUNK_SIZE,
        "chunk_overlap": int(chunk_row[1]) if chunk_row else settings.CHUNK_OVERLAP,
    }


# --- conversations --------------------------------------------------------

async def create_conversation(question: str, answer: str, confidence: float,
                              sources: list[dict], run_id: str = "",
                              trace: dict | None = None) -> dict:
    async with get_session() as session:
        conv = Conversation(
            run_id=run_id or str(uuid.uuid4()), question=question, answer=answer,
            confidence=float(confidence), sources=json.dumps(sources or []),
            trace=json.dumps(trace or {}))
        session.add(conv)
        await session.commit()
        return conv.to_dict()


async def list_conversations(limit: int = 50) -> list[dict]:
    async with get_session() as session:
        rows = (await session.execute(
            select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def get_conversation_by_run(run_id: str) -> dict | None:
    async with get_session() as session:
        row = (await session.execute(
            select(Conversation).where(
                (Conversation.run_id == run_id) | (Conversation.id == run_id))
            .order_by(Conversation.created_at.desc()).limit(1)
        )).scalars().first()
        return row.to_dict() if row else None


# --- ingest jobs ----------------------------------------------------------

async def create_job(kind: str, title: str, source: str,
                     file_data: bytes | None = None,
                     file_name: str = "",
                     meta: dict | None = None) -> dict:
    """Create a queued ingest job. The file bytes ride along in the row
    so the worker can pick the job up even after a restart."""
    async with get_session() as session:
        job = IngestJob(kind=kind, title=title, source=source,
                        status="queued", file_data=file_data,
                        file_name=file_name, meta=json.dumps(meta or {}))
        session.add(job)
        await session.commit()
        return job.to_dict()


async def get_job(job_id: str) -> dict | None:
    async with get_session() as session:
        row = await session.get(IngestJob, job_id)
        return row.to_dict() if row else None


async def get_job_with_bytes(job_id: str) -> tuple[dict | None, bytes | None]:
    """Fetch a job plus its stored upload bytes — what the worker reads
    before it starts extracting text."""
    async with get_session() as session:
        row = await session.get(IngestJob, job_id)
        if row is None:
            return None, None
        return row.to_dict(), row.file_data


async def list_jobs(limit: int = 50) -> list[dict]:
    async with get_session() as session:
        rows = (await session.execute(
            select(IngestJob).order_by(IngestJob.created_at.desc()).limit(limit)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def update_job(job_id: str, **fields) -> dict | None:
    """Patch a job row (status, progress, stats, error, ...). ``stats``
    accepts a dict and is stored as JSON text."""
    async with get_session() as session:
        row = await session.get(IngestJob, job_id)
        if row is None:
            return None
        if "stats" in fields and isinstance(fields["stats"], dict):
            fields["stats"] = json.dumps(fields["stats"])
        if "meta" in fields and isinstance(fields["meta"], dict):
            fields["meta"] = json.dumps(fields["meta"])
        for key, value in fields.items():
            if hasattr(row, key):
                setattr(row, key, value)
        await session.commit()
        return row.to_dict()


async def clear_job_bytes(job_id: str) -> None:
    """Drop the stored upload bytes once the job is finished — the
    chunks already live in Qdrant, so the bytes are no longer needed."""
    async with get_session() as session:
        row = await session.get(IngestJob, job_id)
        if row is not None and row.file_data is not None:
            row.file_data = None
            await session.commit()

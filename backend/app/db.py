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
    # Who owns this document (app/auth_service.py). NULL on rows
    # ingested before auth landed — the first user to register claims
    # them (claim_legacy_data); every later row is stamped at ingest.
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True,
                                                 index=True, default=None)

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
            "access_roles": roles, "meta": meta, "owner_id": self.owner_id,
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
    # Owner of this conversation — see Document.owner_id.
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True,
                                                 index=True, default=None)

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
            "owner_id": self.owner_id,
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
    # Owner of this job — see Document.owner_id. The worker reads it
    # off the row so the finished document gets the same owner.
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True,
                                                 index=True, default=None)

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
            "owner_id": self.owner_id,
        }


# --- auth (see app/auth_service.py) --------------------------------------
# Mirrors Elric's auth tables: one row per account, one row per refresh
# token (only its SHA-256 hash is stored), one row per failed login
# (for the 5-fails / 15-minute lockout).


class User(Base):
    """One login account (email + bcrypt password hash)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RefreshToken(Base):
    """One refresh token. The raw token lives only in the user's
    HttpOnly cookie; here only its SHA-256 hash is stored, so a
    database leak cannot be used to mint sessions. Tokens rotate
    inside one family_id; presenting an already-rotated token revokes
    the whole family (reuse detection)."""

    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    token_hash: Mapped[str] = mapped_column(String(64))
    family_id: Mapped[str] = mapped_column(String(36), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                        nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class LoginAttempt(Base):
    """One failed login, for the lockout (5 fails / 15 minutes)."""

    __tablename__ = "login_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


_engine = None
_session_factory = None


def _normalize_database_url(url: str) -> str:
    """Accept plain Postgres URLs and rewrite them to the async driver
    scheme SQLAlchemy needs. Pratik pastes the URL as ``postgresql://``
    (Supabase's own form); ``create_async_engine`` refuses that and
    needs ``postgresql+asyncpg://``.

    Also drops any query string (``?sslmode=require``,
    ``channel_binding=require``, ...): SQLAlchemy hands those to
    asyncpg as keyword arguments and asyncpg rejects them
    (``TypeError: connect() got an unexpected keyword argument
    'sslmode'`` — that killed the first Neon deploy). TLS is already
    handled separately — ``_get_engine`` passes
    ``connect_args={"ssl": True}`` for non-local hosts — so dropping
    the query string loses nothing."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql+asyncpg://" + url[len(prefix):]
            # Strip the query string only for Postgres URLs. Other
            # schemes (notably sqlite) are returned untouched — a
            # urlparse round-trip would rewrite their slashes.
            return urlparse(url)._replace(query="", fragment="").geturl()
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
    # owner_id columns, added when auth landed (2026-11-24): create_all
    # will not ALTER pre-existing tables, so add them explicitly, each
    # in its OWN transaction for the same Postgres reason as above.
    # The matching indexes let the per-user list queries stay fast.
    for statement in (
        "ALTER TABLE documents ADD COLUMN owner_id VARCHAR(36)",
        "ALTER TABLE conversations ADD COLUMN owner_id VARCHAR(36)",
        "ALTER TABLE ingest_jobs ADD COLUMN owner_id VARCHAR(36)",
        "CREATE INDEX IF NOT EXISTS ix_documents_owner_id ON documents (owner_id)",
        "CREATE INDEX IF NOT EXISTS ix_conversations_owner_id ON conversations (owner_id)",
        "CREATE INDEX IF NOT EXISTS ix_ingest_jobs_owner_id ON ingest_jobs (owner_id)",
    ):
        try:
            async with _engine.begin() as conn:
                await conn.execute(text(statement))
        except Exception:  # noqa: BLE001 - column/index already present
            pass
    logger.info("DB initialised (%s)", settings.DATABASE_URL.split("://")[0])


# --- documents -----------------------------------------------------------

async def create_document(title: str, source: str, text: str, chunk_count: int,
                          chunk_size: int, chunk_overlap: int,
                          meta: dict | None = None,
                          owner_id: str | None = None) -> dict:
    chars = len(text)
    async with get_session() as session:
        doc = Document(title=title, source=source, chars=chars,
                       approx_tokens=estimate_tokens(text), chunk_count=chunk_count,
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                       meta=json.dumps(meta or {}), owner_id=owner_id)
        session.add(doc)
        await session.commit()
        return doc.to_dict()


async def list_documents(*, owner_id: str, limit: int = 100) -> list[dict]:
    """Documents belonging to one user only. owner_id is required —
    there is no unfiltered list anymore (see get_corpus_stats for the
    public totals used by /health)."""
    async with get_session() as session:
        rows = (await session.execute(
            select(Document).where(Document.owner_id == owner_id)
            .order_by(Document.created_at.desc()).limit(limit)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def get_document(doc_id: str, *, owner_id: str) -> dict | None:
    """One document, only when it belongs to this user. A document
    owned by someone else looks exactly like a missing one (None) —
    the caller turns that into a 404 and never reveals it exists."""
    async with get_session() as session:
        row = (await session.execute(
            select(Document).where(Document.id == doc_id,
                                  Document.owner_id == owner_id)
        )).scalars().first()
        return row.to_dict() if row else None


async def delete_document(doc_id: str, *, owner_id: str) -> bool:
    async with get_session() as session:
        row = (await session.execute(
            select(Document).where(Document.id == doc_id,
                                  Document.owner_id == owner_id)
        )).scalars().first()
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
        return True


async def get_corpus_stats(owner_id: str | None = None) -> dict:
    """Corpus totals for run traces: docs, chars, approx tokens, chunks.
    With owner_id the totals are that user's only; without it (the
    public /health endpoint) they cover the whole corpus."""
    async with get_session() as session:
        query = select(
            func.count(Document.id),
            func.coalesce(func.sum(Document.chars), 0),
            func.coalesce(func.sum(Document.approx_tokens), 0),
            func.coalesce(func.sum(Document.chunk_count), 0),
        )
        if owner_id is not None:
            query = query.where(Document.owner_id == owner_id)
        row = (await session.execute(query)).one()
        latest = select(Document.chunk_size, Document.chunk_overlap)
        if owner_id is not None:
            latest = latest.where(Document.owner_id == owner_id)
        chunk_row = (await session.execute(
            latest.order_by(Document.created_at.desc()).limit(1))).first()
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
                              trace: dict | None = None,
                              owner_id: str | None = None) -> dict:
    async with get_session() as session:
        conv = Conversation(
            run_id=run_id or str(uuid.uuid4()), question=question, answer=answer,
            confidence=float(confidence), sources=json.dumps(sources or []),
            trace=json.dumps(trace or {}), owner_id=owner_id)
        session.add(conv)
        await session.commit()
        return conv.to_dict()


async def list_conversations(*, owner_id: str, limit: int = 50) -> list[dict]:
    async with get_session() as session:
        rows = (await session.execute(
            select(Conversation).where(Conversation.owner_id == owner_id)
            .order_by(Conversation.created_at.desc()).limit(limit)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def get_conversation_by_run(run_id: str, *, owner_id: str) -> dict | None:
    async with get_session() as session:
        row = (await session.execute(
            select(Conversation).where(
                ((Conversation.run_id == run_id) | (Conversation.id == run_id))
                & (Conversation.owner_id == owner_id))
            .order_by(Conversation.created_at.desc()).limit(1)
        )).scalars().first()
        return row.to_dict() if row else None


# --- ingest jobs ----------------------------------------------------------

async def create_job(kind: str, title: str, source: str,
                     file_data: bytes | None = None,
                     file_name: str = "",
                     meta: dict | None = None,
                     *, owner_id: str) -> dict:
    """Create a queued ingest job. The file bytes ride along in the row
    so the worker can pick the job up even after a restart. owner_id is
    required: the worker stamps the finished document with it."""
    async with get_session() as session:
        job = IngestJob(kind=kind, title=title, source=source,
                        status="queued", file_data=file_data,
                        file_name=file_name, meta=json.dumps(meta or {}),
                        owner_id=owner_id)
        session.add(job)
        await session.commit()
        return job.to_dict()


async def get_job(job_id: str, *, owner_id: str) -> dict | None:
    """One job, only when it belongs to this user (same 404 rule as
    get_document: another user's job looks missing)."""
    async with get_session() as session:
        row = (await session.execute(
            select(IngestJob).where(IngestJob.id == job_id,
                                    IngestJob.owner_id == owner_id)
        )).scalars().first()
        return row.to_dict() if row else None


async def get_job_with_bytes(job_id: str) -> tuple[dict | None, bytes | None]:
    """Fetch a job plus its stored upload bytes — what the worker reads
    before it starts extracting text."""
    async with get_session() as session:
        row = await session.get(IngestJob, job_id)
        if row is None:
            return None, None
        return row.to_dict(), row.file_data


async def list_jobs(*, owner_id: str, limit: int = 50) -> list[dict]:
    async with get_session() as session:
        rows = (await session.execute(
            select(IngestJob).where(IngestJob.owner_id == owner_id)
            .order_by(IngestJob.created_at.desc()).limit(limit)
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

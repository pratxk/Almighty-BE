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

from sqlalchemy import String, Text, Float, Integer, DateTime, select, func, text
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


_engine = None
_session_factory = None


def _get_engine():
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(settings.DATABASE_URL, future=True)
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
        # explicitly. The try/except makes this idempotent (fresh DBs
        # already have the column; the ALTER then fails harmlessly).
        try:
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

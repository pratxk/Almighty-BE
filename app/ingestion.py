"""Ingestion: PDF / txt / md / URL / raw text -> chunks -> DB + vectors.

Every ingest returns transparency stats — chars, approx tokens
(chars/4), chunks created, chunk size and overlap used — which the API
echoes in ingest responses and ``GET /api/documents``. Chunking is a
simple RecursiveCharacterTextSplitter-style splitter (1000 chars,
200 overlap by default); ``langchain_text_splitters`` is used when
installed, otherwise the built-in splitter below.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from app.config import settings
from app.db import create_document
from app.embeddings import embed_texts, estimate_tokens
from app.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks (default 1000 chars / 200 overlap).

    Prefers langchain_text_splitters when available; the built-in
    splitter below splits on paragraph/sentence boundaries first so
    chunks rarely cut mid-sentence.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""])
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        pass
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:  # try to end on a sentence/paragraph boundary
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + chunk_size // 2, end)
                if cut > start:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def extract_text_from_pdf(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text_from_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")


def extract_text_from_html(html_text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _stats(text: str, chunks: list[str], chunk_size: int, chunk_overlap: int) -> dict:
    return {"chars": len(text), "approx_tokens": estimate_tokens(text),
            "chunks": len(chunks), "chunk_count": len(chunks),
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}


async def ingest_text(title: str, text: str, source: str,
                      chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP,
                      access_roles=None, on_progress=None,
                      owner_id: str | None = None) -> dict:
    """Chunk + embed + store text. Returns {document_id, ..., stats}.

    ``access_roles`` tags the document for guardrails (e.g. ["exec"]):
    only those roles may retrieve its chunks. None means public
    (["employee"]) — see guardrails.normalize_roles.

    ``owner_id`` is the user the document belongs to: it is stamped
    on the document row and on every Qdrant point, so only that user
    can list, retrieve, or delete it. None leaves the document
    ownerless (visible to nobody — only the first-registration claim
    can adopt it).

    ``on_progress`` is an optional async callback the background worker
    passes in: ``await on_progress(stage, done_chunks, total_chunks)``
    at each stage ("chunked", "embedded", "stored"), so the job row can
    report real progress. It is never called for a direct (sync) ingest.
    """
    from app.guardrails import normalize_roles

    roles = normalize_roles(access_roles)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No text could be extracted to ingest.")
    if on_progress is not None:
        await on_progress("chunked", 0, len(chunks))
    # Embed BEFORE creating the DB row, and verify the dimension: the
    # embedding function can silently fall back (e.g. Gemini 429 ->
    # 384-dim hash) while the Qdrant collection was created at the
    # configured dim (768). Upserting mismatched vectors fails with a
    # raw Qdrant 400 *after* the document row was written, leaving
    # docs in the DB/BM25 with no vectors. Fail fast instead.
    vectors = embed_texts(chunks)
    if on_progress is not None:
        await on_progress("embedded", len(chunks), len(chunks))
    store = get_vector_store()
    if store.mode == "qdrant" and vectors and len(vectors[0]) != store.dim:
        raise ValueError(
            f"Embedding dimension mismatch: vectors are {len(vectors[0])}-dim "
            f"but the Qdrant collection '{settings.QDRANT_COLLECTION}' is "
            f"{store.dim}-dim. The embedding provider probably fell back "
            f"(e.g. Gemini rate-limited); fix the provider instead of "
            f"ingesting mismatched vectors.")
    doc = await create_document(title=title, source=source, text=text,
                                chunk_count=len(chunks),
                                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                meta={"access_roles": roles},
                                owner_id=owner_id)
    store.upsert_chunks(
        [{"text": c, "source": source, "title": title, "chunk_index": i,
          "document_id": doc["id"], "metadata": {"access_roles": roles}}
         for i, c in enumerate(chunks)],
        vectors, owner_id=owner_id)
    if on_progress is not None:
        await on_progress("stored", len(chunks), len(chunks))
    logger.info("Ingested %s: %d chunks, %d chars (~%d tokens) from %s "
                "(access_roles=%s)",
                title, len(chunks), len(text), estimate_tokens(text), source,
                roles)
    return {"document_id": doc["id"], "title": title, "source": source,
            "access_roles": roles,
            **_stats(text, chunks, chunk_size, chunk_overlap)}


async def ingest_file(path: str | Path, source_name: str | None = None,
                      chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP,
                      access_roles=None) -> dict:
    path = Path(path)
    text = extract_text_from_file(path)
    # Title comes from the original uploaded filename, not the temp file.
    title = Path(source_name).stem if source_name else path.stem
    return await ingest_text(title=title, text=text,
                             source=source_name or str(path),
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                             access_roles=access_roles)


async def fetch_url_text(url: str) -> tuple[str, str]:
    """Fetch a URL and return (title, plain text). Split out of
    ``ingest_url`` so the background worker can fetch first and then
    run the (progress-reporting) ingest step itself."""
    from app.http import make_async_client

    async with make_async_client(30.0, base_url=url, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "adaptive-rag/1.0"})
        resp.raise_for_status()
        raw = resp.text
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw,
                            flags=re.DOTALL | re.IGNORECASE)
    title = html.unescape(title_match.group(1)).strip() if title_match else url
    return title, extract_text_from_html(raw)


async def ingest_url(url: str, chunk_size: int = CHUNK_SIZE,
                     chunk_overlap: int = CHUNK_OVERLAP,
                     access_roles=None) -> dict:
    title, text = await fetch_url_text(url)
    return await ingest_text(title=title, text=text,
                             source=url, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap,
                             access_roles=access_roles)

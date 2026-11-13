"""Vector store: Qdrant (dense) + BM25 (lexical), fused with RRF.

Modes, logged at startup:
  * ``qdrant`` — real Qdrant at ``QDRANT_URL``; the collection is created
    with ``settings.embedding_dim`` (768 Gemini / 384 MiniLM|hash) cosine.
  * ``memory`` — graceful degradation when Qdrant is unreachable: chunks
    + embeddings are kept in a plain in-memory list, so the API still
    works locally with zero setup.

Hybrid search: dense top-k and BM25 top-k are merged by reciprocal rank
fusion (RRF, k=60), then the merged set is reranked by cosine similarity
to the query embedding (a tiny cosine reranker; scikit-learn is not
needed for this — plain numpy cosine is exact and dependency-light).

BM25 corpus is persisted to ``backend/data/bm25_index.json`` (chunk texts
+ payloads) and rebuilt on startup, so lexical search survives restarts
even in memory mode.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import numpy as np

from app.config import settings
from app.embeddings import cosine_similarity, get_embedding_function

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
BM25_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_index.json"


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """Small persistent BM25 index over ingested chunks.

    Uses ``rank_bm25.BM25Okapi`` when installed; otherwise a plain
    TF-overlap scorer with the same interface, so ingestion never fails
    on a missing optional dependency.
    """

    def __init__(self, path: Path = BM25_PATH):
        self.path = path
        self.chunks: list[dict] = []  # each: {id, text, title, source, ...}
        self._bm25 = None
        self._load()

    def _build(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokens(c["text"]) for c in self.chunks])
        except Exception:
            self._bm25 = None  # TF fallback in search()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self.chunks = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("BM25: loaded %d chunks from %s", len(self.chunks), self.path)
        except Exception as exc:
            logger.warning("BM25: could not load index (%s); starting empty.", exc)
            self.chunks = []
        self._build()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.chunks), encoding="utf-8")
        except Exception as exc:
            logger.warning("BM25: could not persist index (%s).", exc)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)
        self._build()
        self._save()

    def clear(self) -> None:
        """Drop every chunk (used on startup when the metadata DB is empty,
        so a fresh DB can never see chunks from a deleted database)."""
        self.chunks = []
        self._build()
        self._save()

    def remove_by_document(self, document_id: str) -> None:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("document_id") != document_id]
        if len(self.chunks) != before:
            self._build()
            self._save()

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return [(chunk_index, bm25_score)] sorted by score desc."""
        if not self.chunks:
            return []
        qtokens = _tokens(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(qtokens)
            order = np.argsort(-np.asarray(scores, dtype=float))[:top_k]
            return [(int(i), float(scores[i])) for i in order if float(scores[i]) > 0]
        # TF-overlap fallback.
        scored = []
        for i, chunk in enumerate(self.chunks):
            ct = set(_tokens(chunk["text"]))
            overlap = len(set(qtokens) & ct)
            if overlap:
                scored.append((i, float(overlap)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class VectorStore:
    """Qdrant-or-memory vector store with hybrid (dense + BM25) search."""

    def __init__(self):
        self.mode = "memory"
        self.dim = settings.embedding_dim
        self._client = None
        self._memory: list[dict] = []  # {id, vector, payload}
        self.bm25 = BM25Store()
        self._connect()

    # -- setup ------------------------------------------------------------
    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None,
                                  timeout=settings.QDRANT_TIMEOUT)
            client.get_collections()  # connectivity probe
            self.ensure_collection(client, VectorParams(size=self.dim, distance=Distance.COSINE))
            self._client = client
            self.mode = "qdrant"
            logger.info("VectorStore: Qdrant active (%s, collection=%s, dim=%d)",
                        settings.QDRANT_URL, settings.QDRANT_COLLECTION, self.dim)
        except Exception as exc:
            logger.warning(
                "VectorStore: Qdrant unreachable (%s); using in-memory store. "
                "Start Qdrant (docker compose up) for persistence.", exc)
            self._client = None
            self.mode = "memory"
            logger.info("VectorStore: in-memory store active (dim=%d)", self.dim)

    def ensure_collection(self, client, params) -> None:
        from qdrant_client.models import VectorParams  # noqa: F401  (re-export guard)

        try:
            client.get_collection(settings.QDRANT_COLLECTION)
        except Exception:
            client.create_collection(collection_name=settings.QDRANT_COLLECTION,
                                     vectors_config=params)
        # Filtered deletes (delete_by_document) match on the
        # ``document_id`` payload field; Qdrant rejects filtered
        # deletes without a payload index on that field (400). The
        # index is idempotent — creating it when it already exists
        # is a no-op — so this is safe on every startup.
        try:
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="document_id", field_schema=PayloadSchemaType.KEYWORD)
        except Exception as exc:  # noqa: BLE001 - index may already exist
            logger.debug("VectorStore: document_id payload index: %s", exc)
        # Role filtering (guardrails) matches on ``access_roles``; Qdrant
        # Cloud rejects a filtered query without a payload index on that
        # field (400 "Index required but not found"). Same idempotent
        # pattern as document_id above.
        try:
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="access_roles", field_schema=PayloadSchemaType.KEYWORD)
        except Exception as exc:  # noqa: BLE001 - index may already exist
            logger.debug("VectorStore: access_roles payload index: %s", exc)

    def status(self) -> dict:
        return {"store": self.mode, "qdrant_url": settings.QDRANT_URL,
                "collection": settings.QDRANT_COLLECTION, "dim": self.dim,
                "chunks": len(self._memory) if self.mode == "memory" else len(self.bm25.chunks)}

    # -- writes -----------------------------------------------------------
    def upsert_chunks(self, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
        """chunks: [{text, source, title, chunk_index, document_id, metadata}]."""
        # Defense in depth (see ingestion.ingest_text): never send vectors
        # whose dimension differs from the collection's — Qdrant rejects
        # them with a bare 400 and the caller may already have written
        # the document row. In memory mode any dim works (vectors are
        # stored as-is and searched with the same function).
        if self.mode == "qdrant" and vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: vectors are {len(vectors[0])}-dim "
                f"but collection '{settings.QDRANT_COLLECTION}' is {self.dim}-dim.")
        ids = [str(uuid.uuid4()) for _ in chunks]
        payloads = [{
            "text": c["text"], "source": c.get("source", ""), "title": c.get("title", ""),
            "chunk_index": c.get("chunk_index", i), "document_id": c.get("document_id", ""),
            "metadata": c.get("metadata", {}),
            # Guardrails: access roles are a TOP-LEVEL payload field so
            # Qdrant can filter on them server-side; the same list rides
            # in metadata for the BM25 copy (same dicts). Untagged chunks
            # default to public (["employee"]) — see guardrails.py.
            "access_roles": (c.get("metadata", {}) or {}).get("access_roles") or ["employee"],
        } for i, c in enumerate(chunks)]
        if self.mode == "qdrant":
            from qdrant_client.models import PointStruct

            self._client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[PointStruct(id=cid, vector=vec, payload=pl)
                        for cid, vec, pl in zip(ids, vectors, payloads)],
            )
        else:
            self._memory.extend({"id": cid, "vector": vec, "payload": pl}
                                for cid, vec, pl in zip(ids, vectors, payloads))
        self.bm25.add_chunks(payloads)
        return ids

    def delete_by_document(self, document_id: str) -> None:
        if self.mode == "qdrant":
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            self._client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=Filter(must=[FieldCondition(
                    key="document_id", match=MatchValue(value=document_id))]),
            )
        else:
            self._memory = [m for m in self._memory
                            if m["payload"].get("document_id") != document_id]
        self.bm25.remove_by_document(document_id)

    # -- hybrid search ----------------------------------------------------
    def search(self, query: str, query_embedding: list[float], top_k: int,
               role: str | None = None) -> list[dict]:
        """Hybrid dense + BM25 search, RRF fusion, cosine rerank.

        Returns [{text, source, title, chunk_index, document_id, score,
        dense_score, bm25_score}] sorted by final cosine score desc.

        Guardrails: ``role`` filters chunks to what the requester may
        see, BEFORE ranking or generation. Two layers: a Qdrant-side
        ``query_filter`` (restricted points never leave the server) and
        a pool-level filter (covers the BM25 path and legacy chunks with
        no ``access_roles`` field, which are treated as public). Both
        must agree — the pool filter is the backstop.
        """
        from app.guardrails import chunk_access_roles, role_can_access
        from app.config import settings as _settings

        role = (role or "").strip().lower() or _settings.GUARDRAIL_DEFAULT_ROLE
        # 1. Dense candidates (oversample for the fusion pool).
        dense_ranked: list[dict] = []
        if self.mode == "qdrant":
            from qdrant_client.models import (FieldCondition, Filter,
                                              IsEmptyCondition, MatchAny,
                                              PayloadField)

            qfilter = Filter(should=[
                FieldCondition(key="access_roles",
                               match=MatchAny(any=[role, "employee"])),
                # Legacy chunks ingested before roles existed carry no
                # access_roles field; they are public, not invisible.
                IsEmptyCondition(is_empty=PayloadField(key="access_roles")),
            ])
            # NB: qdrant-client >= 1.11 removed client.search(); the
            # current call is query_points (same ScoredPoint shape).
            resp = self._client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_embedding, limit=max(top_k * 2, 10),
                with_payload=True, query_filter=qfilter)
            for h in resp.points:
                dense_ranked.append({**h.payload, "score": float(h.score),
                                     "dense_score": float(h.score)})
        else:
            scored = sorted(
                ((cosine_similarity(query_embedding, m["vector"]), m) for m in self._memory),
                key=lambda x: -x[0])
            for score, m in scored[: max(top_k * 2, 10)]:
                dense_ranked.append({**m["payload"], "score": float(score),
                                     "dense_score": float(score)})

        # 2. BM25 candidates.
        bm25_hits = self.bm25.search(query, max(top_k * 2, 10))
        bm25_by_id: dict[str, float] = {}
        bm25_ranked: list[dict] = []
        for idx, bscore in bm25_hits:
            chunk = self.bm25.chunks[idx]
            key = f"{chunk.get('document_id')}:{chunk.get('chunk_index')}"
            bm25_by_id[key] = bscore
            bm25_ranked.append({**chunk, "bm25_score": bscore})

        # 3. RRF fusion over chunk keys, then cosine rerank of the pool.
        def _key(d: dict) -> str:
            return f"{d.get('document_id')}:{d.get('chunk_index')}"

        pool: dict[str, dict] = {}
        for d in dense_ranked:
            pool[_key(d)] = {**d, "bm25_score": bm25_by_id.get(_key(d), 0.0)}
        for d in bm25_ranked:
            pool.setdefault(_key(d), {**d, "dense_score": 0.0,
                                      "score": 0.0}).setdefault("bm25_score", d["bm25_score"])
            pool[_key(d)]["bm25_score"] = d["bm25_score"]
        # Guardrail backstop: drop any chunk the role may not see BEFORE
        # fusion/ranking — this is what enforces access on the BM25 path
        # (no server-side filter there) and on legacy untagged chunks.
        pool = {k: d for k, d in pool.items()
                if role_can_access(chunk_access_roles(d), role)}
        dense_ranked = [d for d in dense_ranked
                        if _key(d) in pool]
        bm25_ranked = [d for d in bm25_ranked
                       if _key(d) in pool]
        fused = _rrf([[ _key(d) for d in dense_ranked], [_key(d) for d in bm25_ranked]])
        # Cosine rerank: needs the query embedding vs each doc's vector.
        # BM25-only docs have no vector in the pool, so embed them on the fly
        # only when they made the fusion pool (small set, cheap).
        embed_fn = get_embedding_function()
        results: list[dict] = []
        for key, rrf_score in sorted(fused.items(), key=lambda kv: -kv[1]):
            doc = pool[key]
            vec = None
            if self.mode == "memory":
                match = next((m for m in self._memory
                              if _key(m["payload"]) == key), None)
                vec = match["vector"] if match else None
            if vec is None:
                vec = embed_fn.embed_query(doc["text"])
            cos = cosine_similarity(query_embedding, vec)
            results.append({**doc, "score": cos, "dense_score": doc.get("dense_score", cos),
                            "bm25_score": doc.get("bm25_score", 0.0),
                            "rrf_score": rrf_score})
        results.sort(key=lambda d: -d["score"])
        return results[:top_k]


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

"""Auth + isolation test rig.

Real local Postgres (port 5433, database ``almighty_test``) + real
local Redis (port 6379, DB 1) + the LIVE Qdrant collection. The live
collection is used on purpose (there is no local Qdrant): test
documents are titled ``test-isolation-*`` and are DELETED afterwards,
and the point count is verified back to its pre-test value. The
first-registration legacy claim also runs against the live collection
— its effect is RESTORED in the session teardown (owner_id removed
from exactly the points that were ownerless before), and the restore
is verified.

Env is set here BEFORE any app import: config reads .env at import,
but explicit env vars win over the .env file, so DATABASE_URL /
REDIS_URL / JWT_SECRET below take effect while GEMINI_API_KEY /
QDRANT_URL / QDRANT_API_KEY still come from backend/.env.

Run from ``backend/``: ``.venv/bin/python -m pytest tests/ -v``
"""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

os.environ["DATABASE_URL"] = (
    "postgresql+asyncpg://postgres@127.0.0.1:5433/almighty_test")
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/1"
os.environ["JWT_SECRET"] = "test-only-jwt-secret-not-for-production-use"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import (  # noqa: E402
    Conversation, Document, IngestJob, LoginAttempt, RefreshToken, User,
    get_session)
from app.main import app  # noqa: E402
from app.vectorstore import get_vector_store  # noqa: E402

TEST_PASSWORD = "TestPass123!x"  # meets the policy (upper/lower/digit/symbol)
BM25_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_index.json"


def unique_email(tag: str) -> str:
    return f"test-{tag}-{uuid.uuid4().hex[:8]}@example.com"


def register_user(client: TestClient, tag: str) -> dict:
    """Register a fresh user via the API. Returns the user dict plus
    the access token and the raw refresh cookie for later calls."""
    email = unique_email(tag)
    resp = client.post("/api/auth/register",
                       json={"email": email, "password": TEST_PASSWORD})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    user = {
        "id": body["user"]["id"], "email": email,
        "password": TEST_PASSWORD,
        "token": body["access_token"],
        "refresh": resp.cookies.get("almighty_refresh"),
    }
    # The shared TestClient keeps cookies in a jar across requests —
    # clear it so later per-request cookies= are unambiguous (a stale
    # jar cookie would otherwise ride along on "no cookie" calls).
    client.cookies.clear()
    return user


def login_user(client: TestClient, email: str, password: str) -> dict:
    resp = client.post("/api/auth/login",
                       json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user = {
        "id": body["user"]["id"], "email": email, "password": password,
        "token": body["access_token"],
        "refresh": resp.cookies.get("almighty_refresh"),
    }
    client.cookies.clear()
    return user


def auth_headers(user: dict) -> dict:
    return {"Authorization": f"Bearer {user['token']}"}


# -- live Qdrant helpers (the collection under test) ------------------------

def qdrant():
    store = get_vector_store()
    assert store.mode == "qdrant", "tests need the live Qdrant collection"
    return store._client


def count_points() -> int:
    return qdrant().count(
        collection_name=settings.QDRANT_COLLECTION, exact=True).count


def ownerless_point_ids() -> list:
    """Ids of every point whose payload has no owner_id."""
    client = qdrant()
    ids: list = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION, limit=100,
            offset=offset, with_payload=True, with_vectors=False)
        ids.extend(p.id for p in points
                   if (p.payload or {}).get("owner_id") is None)
        if next_offset is None:
            break
        offset = next_offset
    return ids


def points_owned_by(owner_id: str) -> int:
    """How many points carry this owner_id (scroll + count locally —
    the collection is a few hundred points)."""
    client = qdrant()
    owned = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION, limit=100,
            offset=offset, with_payload=True, with_vectors=False)
        owned += sum(1 for p in points
                     if (p.payload or {}).get("owner_id") == owner_id)
        if next_offset is None:
            break
        offset = next_offset
    return owned


# -- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """The app, booted against the test Postgres + test Redis."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_cookie_jar(client):
    """The shared TestClient keeps cookies in a jar across requests, and
    every /login or /refresh response plants a fresh almighty_refresh
    cookie there. A stale jar cookie would ride along on later calls
    that pass their own cookies= (or on 'no cookie' calls) and make
    refresh/logout tests pass or fail for the wrong reason. Clear the
    jar around every test so each test sends exactly the cookies it
    means to send."""
    client.cookies.clear()
    yield
    client.cookies.clear()


async def _reset_test_db() -> None:
    """Empty the test database (auth + data tables). The test DB is
    disposable; resetting it here makes the session's first
    registration always the first user, even across pytest runs."""
    from sqlalchemy import delete

    async with get_session() as session:
        for model in (RefreshToken, LoginAttempt, User,
                      Document, Conversation, IngestJob):
            await session.execute(delete(model))
        await session.commit()


async def _seed_ownerless_rows() -> dict:
    """Ownerless rows in the TEST database, standing in for the
    pre-auth corpus (the real Nablon rows live in Neon, not here).
    Returns their ids."""
    ids = {}
    async with get_session() as session:
        doc = Document(title="test-isolation-legacy-doc", source="test",
                       chars=10, approx_tokens=3, chunk_count=1,
                       owner_id=None)
        conv = Conversation(run_id=str(uuid.uuid4()), question="legacy?",
                            answer="legacy.", owner_id=None)
        job = IngestJob(kind="file", title="test-isolation-legacy-job",
                        source="test", status="done", owner_id=None)
        session.add_all([doc, conv, job])
        await session.commit()
        ids = {"document": doc.id, "conversation": conv.id, "job": job.id}
    return ids


async def _row_owners(ids: dict) -> dict:
    async with get_session() as session:
        out = {}
        for key, model in (("document", Document),
                           ("conversation", Conversation),
                           ("job", IngestJob)):
            row = await session.get(model, ids[key])
            out[key] = row.owner_id if row else "MISSING"
        return out


async def _delete_rows(ids: dict) -> None:
    async with get_session() as session:
        for key, model in (("document", Document),
                           ("conversation", Conversation),
                           ("job", IngestJob)):
            row = await session.get(model, ids[key])
            if row is not None:
                await session.delete(row)
        await session.commit()


def run_in_app_loop(client, coro_fn, *args):
    """Run an async DB helper in the TestClient's event loop.

    The app's async engine is created in that loop (lifespan); calling
    the same helpers with asyncio.run() in the test thread would attach
    its connections to a different loop and asyncpg refuses them.
    """
    return client.portal.call(coro_fn, *args)


@pytest.fixture(scope="session")
def rq_worker():
    """A real RQ ingest worker (subprocess) against the test Postgres +
    test Redis, so the ingest test exercises the true enqueue -> worker
    -> done path instead of calling the pipeline in-process."""
    backend_dir = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["DATABASE_URL"] = (
        "postgresql+asyncpg://postgres@127.0.0.1:5433/almighty_test")
    env["REDIS_URL"] = "redis://127.0.0.1:6379/1"
    env["JWT_SECRET"] = "test-only-jwt-secret-not-for-production-use"
    rq_bin = str(Path(sys.executable).parent / "rq")
    log_path = Path("/tmp/almighty-test-worker.log")
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            [rq_bin, "worker", "--url", "redis://127.0.0.1:6379/1",
             "--path", ".", "ingest"],
            cwd=str(backend_dir), env=env,
            stdout=log_file, stderr=subprocess.STDOUT)
    try:
        # Wait for the worker to register itself in Redis (it imports
        # the app first, which can take ~30s on a cold start).
        import redis as redis_lib

        redis_client = redis_lib.Redis.from_url("redis://127.0.0.1:6379/1")
        deadline = time.time() + 120
        while time.time() < deadline:
            if redis_client.smembers("rq:workers"):
                break
            if proc.poll() is not None:
                raise RuntimeError(
                    f"RQ worker exited early; see {log_path}")
            time.sleep(2)
        else:
            raise RuntimeError(
                f"RQ worker did not register in 120s; see {log_path}")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session", autouse=True)
def first_user(client):
    """The FIRST registration in the session — the one that claims the
    ownerless corpus. Runs before any test, so test_auth's users are
    2nd/3rd/... and never trigger the claim.

    Setup: records the pre-test Qdrant state, seeds ownerless rows in
    the test DB, registers user A, and asserts the claim happened
    (Postgres rows + Qdrant points). Teardown: restores the live
    Qdrant points to ownerless, restores the local BM25 file, flushes
    the test Redis DB, and deletes the seeded rows.
    """
    bm25_snapshot = BM25_PATH.read_bytes() if BM25_PATH.exists() else None
    run_in_app_loop(client, _reset_test_db)
    pre_total = count_points()
    pre_ownerless = ownerless_point_ids()
    assert pre_ownerless, "expected ownerless (pre-auth) points in the live collection"
    seeded = run_in_app_loop(client, _seed_ownerless_rows)

    user_a = register_user(client, "first")
    owners = run_in_app_loop(client, _row_owners, seeded)
    assert owners == {k: user_a["id"] for k in owners}, \
        f"legacy Postgres rows not claimed by first user: {owners}"
    post_ownerless = ownerless_point_ids()
    assert post_ownerless == [], \
        f"legacy Qdrant points not claimed: {len(post_ownerless)} still ownerless"
    assert points_owned_by(user_a["id"]) == len(pre_ownerless), \
        "claimed point count does not match the pre-test ownerless count"

    claim = {
        "user": user_a,
        "pre_total_points": pre_total,
        "pre_ownerless_ids": pre_ownerless,
        "seeded_rows": seeded,
    }
    yield claim

    # -- teardown: put the live collection back exactly as it was. --
    try:
        client_q = qdrant()
        ids = claim["pre_ownerless_ids"]
        for start in range(0, len(ids), 100):
            client_q.delete_payload(
                collection_name=settings.QDRANT_COLLECTION,
                keys=["owner_id"], points=ids[start:start + 100])
        assert set(ownerless_point_ids()) == set(ids), \
            "Qdrant restore failed: ownerless set differs from pre-test"
        assert points_owned_by(user_a["id"]) == 0, \
            "Qdrant restore failed: points still owned by the test user"
        assert count_points() == claim["pre_total_points"], \
            "Qdrant point count differs from pre-test value"
    finally:
        if bm25_snapshot is not None:
            BM25_PATH.write_bytes(bm25_snapshot)
        elif BM25_PATH.exists():
            BM25_PATH.unlink()
        run_in_app_loop(client, _delete_rows, claim["seeded_rows"])
        import redis as redis_lib

        redis_lib.Redis.from_url("redis://127.0.0.1:6379/1").flushdb()


@pytest.fixture(scope="session")
def user_b(client):
    """A second user — registered after the first, so the legacy claim
    does not apply to them: they start with an empty library."""
    return register_user(client, "second")

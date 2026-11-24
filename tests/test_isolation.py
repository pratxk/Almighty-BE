"""Per-user isolation tests — the critical suite.

Two users: A is the FIRST user of the session (registered by the
``first_user`` fixture in conftest.py, so they claimed the ownerless
legacy corpus); B registered after them and starts empty.

Live Qdrant collection: the test document is titled
``test-isolation-*`` and is deleted at the end of the suite; the
point count is verified back to its pre-test value (conftest
teardown). The legacy claim's Qdrant effect is also restored there.
"""
import time
import uuid

from conftest import auth_headers, count_points


def _chat(client, user, question):
    resp = client.post("/api/chat", headers=auth_headers(user),
                       json={"question": question})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _trace(client, user, run_id):
    """The run trace for a chat (retrieved/kept counts are the precise,
    LLM-independent record of what local retrieval found)."""
    resp = client.get(f"/api/runs/{run_id}", headers=auth_headers(user))
    assert resp.status_code == 200, resp.text
    return resp.json()["trace"]


def _wait_for_job_done(client, user, job_id, timeout_s=120):
    """Poll GET /api/jobs/{id} until the real RQ worker finishes it."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}",
                         headers=auth_headers(user)).json()
        if job["status"] == "done":
            return job
        if job["status"] == "failed":
            raise AssertionError(f"ingest job failed: {job}")
        time.sleep(2)
    raise AssertionError(f"ingest job {job_id} not done in {timeout_s}s")


def test_legacy_claim_first_user_only(client, first_user, user_b):
    user_a = first_user["user"]
    # A sees the seeded legacy document; B sees nothing.
    docs_a = client.get("/api/documents",
                        headers=auth_headers(user_a)).json()["documents"]
    assert any(d["title"] == "test-isolation-legacy-doc" for d in docs_a)
    docs_b = client.get("/api/documents",
                        headers=auth_headers(user_b)).json()["documents"]
    assert docs_b == []

    # A can ask about the pre-auth Nablon guide (claimed Qdrant points);
    # B gets no answer grounded in it.
    question = "What is the ShopKart example in the interview guide?"
    answer_a = _chat(client, user_a, question)
    assert "shopkart" in answer_a["answer"].lower(), answer_a["answer"]
    assert any("nablon" in (s.get("title") or "").lower()
               or "nablon" in (s.get("source") or "").lower()
               for s in answer_a["sources"]), answer_a["sources"]
    assert _trace(client, user_a, answer_a["run_id"])["retrieved"] > 0
    answer_b = _chat(client, user_b, question)
    # NB: B's refusal quotes the question, so the word "shopkart" can
    # appear in it — the isolation guarantee is that B's run retrieved
    # NOTHING locally and cites no local source, not the absence of a
    # word that was in the question itself.
    assert _trace(client, user_b, answer_b["run_id"])["retrieved"] == 0
    assert not any(s.get("source_type") == "local"
                   for s in answer_b["sources"]), answer_b["sources"]


def test_zebra_isolation(client, first_user, user_b, rq_worker):
    user_a = first_user["user"]
    marker = f"ZEBRA-ISOLATION-{uuid.uuid4().hex[:6].upper()}"
    text = (
        f"The {marker} is a fictional striped test creature. "
        f"The {marker} lives only in test documents and eats bamboo. "
        f"Researchers study the {marker} to check that answers stay "
        f"inside one user's own documents.")
    question = f"What is the {marker}?"

    # A uploads the document (202); the real RQ worker runs it to done.
    resp = client.post("/api/ingest", headers=auth_headers(user_a),
                       files={"file": ("test-isolation-zebra.txt", text,
                                       "text/plain")})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    job = _wait_for_job_done(client, user_a, job_id)
    doc_id = job["stats"]["document_id"]

    # Everything below runs in a try/finally so that a failing assert
    # never leaves the test document (and its Qdrant points) behind for
    # the fixture teardown to trip over.
    try:
        # A sees the document; A gets a cited answer with the marker.
        docs_a = client.get("/api/documents",
                            headers=auth_headers(user_a)).json()["documents"]
        assert any(d["id"] == doc_id for d in docs_a)
        answer_a = _chat(client, user_a, question)
        assert marker.lower() in answer_a["answer"].lower(), answer_a["answer"]
        assert any(s.get("source_type", "local") == "local"
                   for s in answer_a["sources"]), answer_a["sources"]
        assert _trace(client, user_a, answer_a["run_id"])["retrieved"] > 0
        run_id = answer_a["run_id"]

        # B sees nothing: empty library, no answer with the marker.
        docs_b = client.get("/api/documents",
                            headers=auth_headers(user_b)).json()["documents"]
        assert docs_b == []
        answer_b = _chat(client, user_b, question)
        # NB: B's refusal quotes the question, so the marker token can
        # appear in it — the isolation guarantee is that B's run
        # retrieved NOTHING locally, cites no local source, and repeats
        # none of the document's facts ("bamboo" is in the doc only,
        # never in the question).
        assert _trace(client, user_b, answer_b["run_id"])["retrieved"] == 0
        assert not any(s.get("source_type") == "local"
                       for s in answer_b["sources"]), answer_b["sources"]
        assert "bamboo" not in answer_b["answer"].lower(), answer_b["answer"]

        # B cannot reach A's document or job by id (404, never 403).
        assert client.get(f"/api/documents/{doc_id}",
                          headers=auth_headers(user_b)).status_code == 404
        assert client.delete(f"/api/documents/{doc_id}",
                             headers=auth_headers(user_b)).status_code == 404
        assert client.get(f"/api/jobs/{job_id}",
                          headers=auth_headers(user_b)).status_code == 404
        assert client.get("/api/jobs",
                          headers=auth_headers(user_b)).json() == {"jobs": []}
        # A's document survived B's delete attempt.
        assert client.get(f"/api/documents/{doc_id}",
                          headers=auth_headers(user_a)).status_code == 200
        assert client.get(f"/api/jobs/{job_id}",
                          headers=auth_headers(user_a)).status_code == 200

        # Cache: A asking again is a cache hit; B asking again still gets
        # no leaked answer (and is not served A's cached answer).
        again_a = _chat(client, user_a, question)
        assert again_a.get("cached") is True
        assert marker.lower() in again_a["answer"].lower()
        again_b = _chat(client, user_b, question)
        assert again_b.get("cached") is not True
        assert "bamboo" not in again_b["answer"].lower(), again_b["answer"]
        assert not any(s.get("source_type") == "local"
                       for s in again_b["sources"]), again_b["sources"]

        # Run traces: B cannot read A's run; A can.
        assert client.get(f"/api/runs/{run_id}",
                          headers=auth_headers(user_b)).status_code == 404
        assert client.get(f"/api/runs/{run_id}",
                          headers=auth_headers(user_a)).status_code == 200

        # A deletes their own document; the Qdrant point count returns to
        # its pre-test value.
        deleted = client.delete(f"/api/documents/{doc_id}",
                                headers=auth_headers(user_a))
        assert deleted.status_code == 200
        docs_a = client.get("/api/documents",
                            headers=auth_headers(user_a)).json()["documents"]
        assert not any(d["id"] == doc_id for d in docs_a)
        assert count_points() == first_user["pre_total_points"]
    finally:
        # Best-effort cleanup if an assert above failed before A's
        # delete: remove the test document (and its points) as A.
        still_there = client.get(f"/api/documents/{doc_id}",
                                 headers=auth_headers(user_a))
        if still_there.status_code == 200:
            client.delete(f"/api/documents/{doc_id}",
                          headers=auth_headers(user_a))

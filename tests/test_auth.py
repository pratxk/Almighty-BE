"""Auth endpoint tests: register / login / refresh / logout / me /
change-password, lockout, and unauthenticated access to data endpoints.

Each test uses fresh users (uuid emails) so tests do not interfere.
The first registration of the whole session happens in the
``first_user`` fixture (conftest.py) — the legacy claim — so the users
here are 2nd/3rd/... and the claim does not re-fire.
"""
import uuid

from conftest import (
    TEST_PASSWORD, auth_headers, login_user, register_user, unique_email)


def test_register_returns_201_token_and_cookie(client):
    user = register_user(client, "auth")
    assert user["token"]
    assert user["refresh"]
    resp = client.get("/api/auth/me", headers=auth_headers(user))
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == user["email"]


def test_register_duplicate_email_is_409(client):
    email = unique_email("dup")
    resp = client.post("/api/auth/register",
                       json={"email": email, "password": TEST_PASSWORD})
    assert resp.status_code == 201
    resp = client.post("/api/auth/register",
                       json={"email": email, "password": TEST_PASSWORD})
    assert resp.status_code == 409


def test_register_rejects_weak_password_and_bad_email(client):
    resp = client.post("/api/auth/register",
                       json={"email": unique_email("weak"),
                             "password": "short"})
    assert resp.status_code == 400
    resp = client.post("/api/auth/register",
                       json={"email": "not-an-email",
                             "password": TEST_PASSWORD})
    assert resp.status_code == 400


def test_login_ok_wrong_password_unknown_email(client):
    user = register_user(client, "login")
    ok = login_user(client, user["email"], TEST_PASSWORD)
    assert ok["token"]
    wrong = client.post("/api/auth/login",
                        json={"email": user["email"],
                              "password": "WrongPass123!x"})
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid email or password"
    unknown = client.post("/api/auth/login",
                          json={"email": unique_email("ghost"),
                                "password": TEST_PASSWORD})
    assert unknown.status_code == 401
    # Wrong password and unknown email are indistinguishable.
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_lockout_after_five_failed_logins(client):
    email = unique_email("lockout")
    for _ in range(5):
        resp = client.post("/api/auth/login",
                           json={"email": email,
                                 "password": "WrongPass123!x"})
        assert resp.status_code == 401
    # The 6th attempt — even with the right password — is locked out.
    resp = client.post("/api/auth/login",
                       json={"email": email, "password": TEST_PASSWORD})
    assert resp.status_code == 429


def test_refresh_rotates_and_old_token_is_rejected(client):
    user = register_user(client, "refresh")
    old_cookie = user["refresh"]
    resp = client.post("/api/auth/refresh",
                       cookies={"almighty_refresh": old_cookie})
    assert resp.status_code == 200, resp.text
    new_cookie = resp.cookies.get("almighty_refresh")
    assert new_cookie and new_cookie != old_cookie
    assert resp.json()["access_token"]
    # The 200 above planted the new cookie in the jar — clear it so the
    # reuse call below sends ONLY the old (rotated) token.
    client.cookies.clear()
    # Reusing the rotated token is reuse: 401, and the whole family is
    # revoked — so the new token's refresh fails too.
    reuse = client.post("/api/auth/refresh",
                        cookies={"almighty_refresh": old_cookie})
    assert reuse.status_code == 401
    after_reuse = client.post("/api/auth/refresh",
                              cookies={"almighty_refresh": new_cookie})
    assert after_reuse.status_code == 401


def test_refresh_with_unknown_token_is_401_and_revokes_nothing(client):
    user = register_user(client, "unknown-refresh")
    resp = client.post("/api/auth/refresh",
                       cookies={"almighty_refresh": "no-such-token"})
    assert resp.status_code == 401
    # The real token still works afterwards (nothing was revoked).
    ok = client.post("/api/auth/refresh",
                     cookies={"almighty_refresh": user["refresh"]})
    assert ok.status_code == 200


def test_refresh_without_cookie_is_401(client):
    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 401


def test_logout_revokes_token_and_clears_cookie(client):
    user = register_user(client, "logout")
    resp = client.post("/api/auth/logout",
                       cookies={"almighty_refresh": user["refresh"]})
    assert resp.status_code == 200
    set_cookie = resp.headers.get("set-cookie", "")
    assert "almighty_refresh" in set_cookie
    after = client.post("/api/auth/refresh",
                        cookies={"almighty_refresh": user["refresh"]})
    assert after.status_code == 401


def test_me_needs_token(client):
    user = register_user(client, "me")
    ok = client.get("/api/auth/me", headers=auth_headers(user))
    assert ok.status_code == 200
    assert ok.json()["user"]["id"] == user["id"]
    anon = client.get("/api/auth/me")
    assert anon.status_code == 401
    bad = client.get("/api/auth/me",
                     headers={"Authorization": "Bearer not-a-token"})
    assert bad.status_code == 401


def test_change_password_revokes_all_sessions(client):
    user = register_user(client, "changepw")
    new_password = "NewPass456!y"
    wrong = client.post("/api/auth/change-password",
                        headers=auth_headers(user),
                        json={"current_password": "WrongPass123!x",
                              "new_password": new_password})
    assert wrong.status_code == 400
    ok = client.post("/api/auth/change-password",
                     headers=auth_headers(user),
                     json={"current_password": TEST_PASSWORD,
                           "new_password": new_password})
    assert ok.status_code == 200
    # Every refresh token is revoked: the old cookie no longer refreshes.
    gone = client.post("/api/auth/refresh",
                       cookies={"almighty_refresh": user["refresh"]})
    assert gone.status_code == 401
    # The old password is dead, the new one works.
    old_login = client.post("/api/auth/login",
                            json={"email": user["email"],
                                  "password": TEST_PASSWORD})
    assert old_login.status_code == 401
    new_login = login_user(client, user["email"], new_password)
    assert new_login["token"]


def test_data_endpoints_require_auth(client):
    anon = client
    assert anon.get("/api/documents").status_code == 401
    assert anon.get("/api/jobs").status_code == 401
    assert anon.get("/api/conversations").status_code == 401
    assert anon.get("/api/runs/anything").status_code == 401
    assert anon.get("/api/documents/anything").status_code == 401
    assert anon.delete("/api/documents/anything").status_code == 401
    assert anon.post("/api/chat",
                     json={"question": "hello"}).status_code == 401
    # /health stays public.
    assert anon.get("/health").status_code == 200
    # A garbage token is still 401, not 500.
    bad = {"Authorization": "Bearer garbage"}
    assert anon.get("/api/documents", headers=bad).status_code == 401


def test_email_is_normalized_to_lowercase(client):
    email = f"MiXeD-{uuid.uuid4().hex[:6]}@Example.COM"
    resp = client.post("/api/auth/register",
                       json={"email": email, "password": TEST_PASSWORD})
    assert resp.status_code == 201
    assert resp.json()["user"]["email"] == email.lower()

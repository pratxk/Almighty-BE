"""JWT access tokens (HS256, 15-minute expiry).

Mirrors Elric's token contract (invoice-agent backend): the access
token is a short-lived JWT carrying the user id as ``sub``, returned
as ``access_token`` in JSON and sent back as ``Authorization: Bearer``.
Refresh tokens are opaque and live in app/auth_service.py, not here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import jwt as pyjwt

from app.config import settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 15


def require_jwt_secret() -> str:
    """Return the configured JWT secret, or fail loudly.

    Called at startup (see app/main.py lifespan) so a deploy without
    JWT_SECRET refuses to boot instead of issuing tokens nobody can
    verify. Local dev: the value lives in backend/.env (gitignored).
    """
    secret = settings.JWT_SECRET
    if not secret:
        raise RuntimeError(
            "JWT_SECRET is not set. Set it in the environment (Render: "
            "dashboard > Environment) or in backend/.env for local dev."
        )
    return secret


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }
    return pyjwt.encode(payload, require_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Return the user id in a valid access token, else None.

    Expired, tampered, or wrongly-signed tokens all return None —
    the caller answers 401 and never says why.
    """
    try:
        payload = pyjwt.decode(
            token, require_jwt_secret(), algorithms=[ALGORITHM])
    except pyjwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    return user_id if isinstance(user_id, str) and user_id else None

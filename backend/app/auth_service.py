"""User accounts, passwords, refresh tokens, and login lockout.

Mirrors Elric's auth service (invoice-agent backend) — same rules,
adapted to Almighty's async SQLAlchemy sessions (Elric uses sync
sessions; the behaviour is identical, the session calls are async):

- bcrypt cost 12, via the ``bcrypt`` package directly (no passlib).
- Passwords: 12-72 chars, at least one upper, one lower, one digit,
  one symbol.
- Emails are lowercased and validated; login never reveals whether
  an email is registered (wrong password and unknown email both get
  the same 401 message).
- Access tokens are short-lived JWTs (see app/jwt_tokens.py).
  Refresh tokens are opaque random strings, stored only as SHA-256
  hashes, 7-day expiry, rotated inside one family_id; presenting an
  already-rotated or expired token revokes the WHOLE family (reuse
  detection) and forces re-login. An unknown token hash revokes
  nothing, just 401.
- Login lockout: 5 failed attempts in 15 minutes, DB-backed (429).
- The first user to register claims the ownerless corpus (see
  claim_legacy_data): every document, conversation, job, and Qdrant
  point ingested before auth landed becomes theirs. Later users
  start with an empty library.
"""
from __future__ import annotations

import hashlib
import logging
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import delete, func, select, update

from app.db import (
    Conversation,
    Document,
    IngestJob,
    LoginAttempt,
    RefreshToken,
    User,
    get_session,
)
from app.jwt_tokens import create_access_token, decode_access_token

logger = logging.getLogger(__name__)

REFRESH_TOKEN_DAYS = 7
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
BCRYPT_COST = 12

INVALID_LOGIN_MESSAGE = "Invalid email or password"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PASSWORD_RE = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{12,72}$")


class TooManyAttemptsError(Exception):
    """Raised when a login is attempted while the account is locked out."""


def _as_aware(value: datetime) -> datetime:
    """Make a datetime timezone-aware (UTC) if it is naive.

    Postgres returns aware datetimes; SQLite returns naive ones.
    All comparisons in this module use aware datetimes, so normalise
    on read instead of crashing on the SQLite path."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- validation / passwords ----------------------------------------------

def validate_email(email: str) -> str:
    clean = email.strip().lower()
    if not _EMAIL_RE.match(clean) or len(clean) > 320:
        raise ValueError("Enter a valid email address.")
    return clean


def validate_password(password: str) -> None:
    if not _PASSWORD_RE.match(password):
        raise ValueError(
            "Password must be 12-72 characters and include an upper-case "
            "letter, a lower-case letter, a digit, and a symbol.")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_COST)).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def public_user(user: User) -> dict:
    return {"id": user.id, "email": user.email,
            "created_at": (user.created_at.isoformat()
                           if user.created_at else None)}


# --- users ----------------------------------------------------------------

async def get_user(user_id: str) -> User | None:
    async with get_session() as session:
        return await session.get(User, user_id)


async def get_user_by_email(email: str) -> User | None:
    async with get_session() as session:
        result = await session.execute(
            select(User).where(User.email == email))
        return result.scalars().first()


async def count_users() -> int:
    async with get_session() as session:
        result = await session.execute(select(func.count()).select_from(User))
        return result.scalar() or 0


async def create_user(email: str, password: str) -> dict:
    """Register a user. Returns the public user dict.

    Raises ValueError("Email already registered.") on a duplicate —
    the router turns that into a 409.
    """
    clean_email = validate_email(email)
    validate_password(password)
    async with get_session() as session:
        existing = await session.execute(
            select(User).where(User.email == clean_email))
        if existing.scalars().first() is not None:
            raise ValueError("Email already registered.")
        user = User(email=clean_email, password_hash=hash_password(password))
        session.add(user)
        await session.commit()
        await session.refresh(user)
        session.expunge(user)
    # First user claims the ownerless corpus (Nablon guide, sample
    # docs, ...). Later registrations get an empty library.
    if await count_users() == 1:
        await claim_legacy_data(user.id)
    return public_user(user)


async def change_password(user_id: str, current_password: str,
                          new_password: str) -> None:
    """Change a password and revoke ALL refresh tokens, so every other
    session (and the caller's, on its next refresh) must log in again.
    Raises ValueError on a wrong current password or a weak new one."""
    user = await get_user(user_id)
    if user is None or not check_password(current_password, user.password_hash):
        raise ValueError("Current password is incorrect.")
    validate_password(new_password)
    async with get_session() as session:
        db_user = await session.get(User, user_id)
        if db_user is None:
            raise ValueError("Current password is incorrect.")
        db_user.password_hash = hash_password(new_password)
        await session.commit()
    await revoke_all_refresh_tokens(user_id)


# --- legacy claim ----------------------------------------------------------

async def claim_legacy_data(user_id: str) -> dict:
    """Give every ownerless row and Qdrant point to ``user_id``.

    Rule: the FIRST user to register claims the corpus that existed
    before auth (Nablon guide parts, sample docs, ...). It runs at
    registration time, is idempotent (only NULL owner_id rows and
    points without an owner_id payload are touched), and is fast for
    the few hundred points involved. If Qdrant is unreachable the
    Postgres rows are still claimed; the ownerless points stay
    invisible (every search filters owner_id) until the claim is
    re-run — they are never served to the wrong user.
    """
    claimed = {"documents": 0, "conversations": 0, "jobs": 0,
               "qdrant_points": 0}
    async with get_session() as session:
        for model, key in ((Document, "documents"),
                           (Conversation, "conversations"),
                           (IngestJob, "jobs")):
            result = await session.execute(
                update(model).where(model.owner_id.is_(None))
                .values(owner_id=user_id))
            claimed[key] = result.rowcount or 0
        await session.commit()
    try:
        # Lazy import: auth must stay importable even when the vector
        # store cannot connect.
        from app.vectorstore import get_vector_store

        claimed["qdrant_points"] = get_vector_store().claim_ownerless_points(user_id)
    except Exception as exc:  # noqa: BLE001 - Qdrant down: rows are claimed, points wait
        logger.warning("Legacy claim: Qdrant points not claimed (%s). "
                       "Re-run claim_legacy_data once Qdrant is back.", exc)
    logger.info("Legacy claim for user %s: %s", user_id, claimed)
    return claimed


# --- login lockout ---------------------------------------------------------

async def record_failed_login(email: str) -> None:
    async with get_session() as session:
        session.add(LoginAttempt(email=email))
        await session.commit()


async def clear_failed_logins(email: str) -> None:
    async with get_session() as session:
        await session.execute(
            delete(LoginAttempt).where(LoginAttempt.email == email))
        await session.commit()


async def failed_login_count(email: str, since: datetime) -> int:
    async with get_session() as session:
        result = await session.execute(
            select(func.count()).select_from(LoginAttempt).where(
                LoginAttempt.email == email,
                LoginAttempt.created_at >= since))
        return result.scalar() or 0


async def is_locked_out(email: str) -> bool:
    since = _now() - timedelta(minutes=LOCKOUT_MINUTES)
    return await failed_login_count(email, since) >= MAX_LOGIN_ATTEMPTS


async def authenticate(email: str, password: str) -> dict | None:
    """Check credentials. Returns the public user dict on success,
    None on failure — wrong password and unknown email are
    indistinguishable to the caller (same 401 message).

    Raises TooManyAttemptsError when the email is locked out (the
    router turns that into a 429). A successful login clears the
    failed-attempt history for that email.
    """
    clean_email = validate_email(email)
    if await is_locked_out(clean_email):
        raise TooManyAttemptsError()
    user = await get_user_by_email(clean_email)
    if user is None or not check_password(password, user.password_hash):
        await record_failed_login(clean_email)
        return None
    await clear_failed_logins(clean_email)
    return public_user(user)


# --- tokens -----------------------------------------------------------------

def issue_access_token(user: dict) -> str:
    return create_access_token(user["id"])


async def user_from_access_token(token: str) -> dict | None:
    """Return the public user dict for a valid access token, else None."""
    user_id = decode_access_token(token)
    if user_id is None:
        return None
    user = await get_user(user_id)
    return public_user(user) if user else None


def _hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_refresh_token(user_id: str) -> tuple[str, str]:
    """Create a refresh token. Returns (raw_token, family_id).

    The raw token is returned once, for the HttpOnly cookie — only
    its hash is stored."""
    raw = secrets.token_urlsafe(48)
    family_id = str(uuid.uuid4())
    async with get_session() as session:
        session.add(RefreshToken(
            user_id=user_id, token_hash=_hash_refresh_token(raw),
            family_id=family_id,
            expires_at=_now() + timedelta(days=REFRESH_TOKEN_DAYS)))
        await session.commit()
    return raw, family_id


async def rotate_refresh_token(raw_token: str) -> tuple[str, str, str] | None:
    """Rotate a refresh token. Returns (new_raw_token, family_id,
    user_id), or None when the token must not be honoured.

    - Unknown token hash: None, and nothing is revoked (just 401).
    - Already-rotated or expired token: the WHOLE family is revoked
      (reuse detection) and None is returned — the user must log in
      again.
    - Valid token: the old token is revoked and a new token is issued
      in the SAME family.
    """
    token_hash = _hash_refresh_token(raw_token)
    async with get_session() as session:
        result = await session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash))
        row = result.scalars().first()
        if row is None:
            return None
        now = _now()
        if row.revoked_at is not None or _as_aware(row.expires_at) <= now:
            await session.execute(
                update(RefreshToken)
                .where(RefreshToken.family_id == row.family_id)
                .values(revoked_at=now))
            await session.commit()
            return None
        row.revoked_at = now
        new_raw = secrets.token_urlsafe(48)
        session.add(RefreshToken(
            user_id=row.user_id, token_hash=_hash_refresh_token(new_raw),
            family_id=row.family_id,
            expires_at=now + timedelta(days=REFRESH_TOKEN_DAYS)))
        await session.commit()
        return new_raw, row.family_id, row.user_id


async def revoke_refresh_token(raw_token: str) -> None:
    """Logout: revoke just this refresh token."""
    async with get_session() as session:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == _hash_refresh_token(raw_token))
            .values(revoked_at=_now()))
        await session.commit()


async def revoke_all_refresh_tokens(user_id: str) -> None:
    """Revoke every refresh token for a user (password change)."""
    async with get_session() as session:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id)
            .values(revoked_at=_now()))
        await session.commit()

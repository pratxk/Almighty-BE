"""Auth endpoints: register, login, refresh, logout, me, change-password.

Mirrors Elric's auth router (invoice-agent backend): same token
contract, same status codes, same cookie. The refresh token rides in
an HttpOnly cookie named ``almighty_refresh`` (SameSite=Lax, Path=/,
Secure only over HTTPS); the access token is returned as
``access_token`` in JSON and sent back as ``Authorization: Bearer``.

All routes here are public; the data endpoints in app/main.py are
protected with ``Depends(require_user)``.
"""
from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app import auth_service
from app.auth_service import INVALID_LOGIN_MESSAGE, TooManyAttemptsError

router = APIRouter(prefix="/api/auth", tags=["auth"])

REFRESH_COOKIE = "almighty_refresh"
REFRESH_COOKIE_MAX_AGE = 7 * 24 * 60 * 60  # 7 days, matches the token expiry

_bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _set_refresh_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        REFRESH_COOKIE, token,
        max_age=REFRESH_COOKIE_MAX_AGE,
        httponly=True, samesite="lax", path="/",
        secure=request.url.scheme == "https",
    )


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Dependency for protected endpoints: returns the public user
    dict for a valid Bearer access token, else 401."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await auth_service.user_from_access_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.post("/register", status_code=201, response_model=AuthResponse)
async def register(payload: RegisterRequest, request: Request,
                   response: Response) -> AuthResponse:
    try:
        user = await auth_service.create_user(payload.email, payload.password)
    except ValueError as exc:
        detail = str(exc)
        status = 409 if "already registered" in detail.lower() else 400
        raise HTTPException(status_code=status, detail=detail)
    refresh_token, _family_id = await auth_service.create_refresh_token(user["id"])
    _set_refresh_cookie(response, request, refresh_token)
    return AuthResponse(access_token=auth_service.issue_access_token(user),
                        user=user)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, request: Request,
                response: Response) -> AuthResponse:
    try:
        user = await auth_service.authenticate(payload.email, payload.password)
    except TooManyAttemptsError:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Try again in 15 minutes.")
    if user is None:
        raise HTTPException(status_code=401, detail=INVALID_LOGIN_MESSAGE)
    refresh_token, _family_id = await auth_service.create_refresh_token(user["id"])
    _set_refresh_cookie(response, request, refresh_token)
    return AuthResponse(access_token=auth_service.issue_access_token(user),
                        user=user)


@router.post("/refresh", response_model=AuthResponse)
async def refresh(request: Request, response: Response,
                  almighty_refresh: str | None = Cookie(default=None)) -> AuthResponse:
    if not almighty_refresh:
        raise HTTPException(status_code=401, detail="Not authenticated")
    rotated = await auth_service.rotate_refresh_token(almighty_refresh)
    if rotated is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    new_raw, _family_id, user_id = rotated
    user = await auth_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    _set_refresh_cookie(response, request, new_raw)
    return AuthResponse(
        access_token=auth_service.issue_access_token(
            auth_service.public_user(user)),
        user=auth_service.public_user(user))


@router.post("/logout")
async def logout(response: Response,
                 almighty_refresh: str | None = Cookie(default=None)) -> dict:
    if almighty_refresh:
        await auth_service.revoke_refresh_token(almighty_refresh)
    response.delete_cookie(REFRESH_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
async def me(user: dict = Depends(require_user)) -> dict:
    return {"user": user}


@router.post("/change-password")
async def change_password(payload: ChangePasswordRequest,
                          user: dict = Depends(require_user)) -> dict:
    """Change the password. Revokes ALL refresh tokens, so every
    session (including this one, on its next refresh) must log in
    again. The current access token stays valid until it expires
    (15 minutes at most)."""
    try:
        await auth_service.change_password(
            user["id"], payload.current_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}

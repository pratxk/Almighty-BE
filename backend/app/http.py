"""httpx client factory.

Why this exists: ``httpx`` 0.28's environment-proxy parsing breaks on
this machine's ``no_proxy`` value (IPv6 entries such as ``[::1]`` get
mangled into bogus proxy keys and every request then fails with
``InvalidURL: Invalid port: ':1]'``), while ``trust_env=False`` breaks
the opposite way — this VM has no direct internet access, so HTTPS to
a public host without the egress proxy fails with an SSL error.

So we do proxy selection ourselves, explicitly and simply:
  * public hosts  -> proxy from ``HTTPS_PROXY``/``https_proxy`` (env),
  * localhost / hosts listed in ``no_proxy`` -> direct, no proxy.

No proxy credentials are read, stored, or logged here — the raw env
value is passed straight through to httpx.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _env_proxy() -> str | None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _verify() -> str | bool:
    """TLS trust store.

    httpcore defaults to certifi's bundle, which does not contain this
    machine's egress-proxy CA — requests then fail with
    ``CERTIFICATE_VERIFY_FAILED: self-signed certificate``. The VM
    publishes its CA bundle via ``SSL_CERT_FILE`` (also
    ``REQUESTS_CA_BUNDLE``); use it when present, otherwise default
    verification. Never disable verification.
    """
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(name)
        if value:
            return value
    return True


def _no_proxy_hosts() -> list[str]:
    raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _host_needs_proxy(host: str) -> bool:
    host = (host or "").lower().strip("[]")
    if host in _LOCAL_HOSTS:
        return False
    for entry in _no_proxy_hosts():
        entry = entry.strip("[]").lower()
        if not entry:
            continue
        if entry.startswith("*."):
            if host.endswith(entry[1:]):
                return False
        elif host == entry or host.endswith("." + entry):
            return False
    return True


def proxy_for_url(url: str) -> str | None:
    """Proxy URL to use for ``url``, or None for a direct connection."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return _env_proxy()
    return _env_proxy() if _host_needs_proxy(host) else None


def make_client(timeout: float, base_url: str | None = None, **kwargs) -> httpx.Client:
    """Synchronous httpx client with correct proxy handling for this host."""
    proxy = proxy_for_url(base_url) if base_url else _env_proxy()
    return httpx.Client(proxy=proxy, trust_env=False, timeout=timeout,
                        verify=_verify(), **kwargs)


def make_async_client(timeout: float, base_url: str | None = None,
                      **kwargs) -> httpx.AsyncClient:
    """Async httpx client with correct proxy handling for this host."""
    proxy = proxy_for_url(base_url) if base_url else _env_proxy()
    return httpx.AsyncClient(proxy=proxy, trust_env=False, timeout=timeout,
                             verify=_verify(), **kwargs)

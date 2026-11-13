"""Application settings loaded from environment variables.

No secrets are hardcoded here. Copy ``../.env.example`` to ``../.env``
(or export variables) and fill in real values. The app runs with no
keys at all in degraded/demo mode (extractive answers, local/hash
embeddings); it never fakes LLM generation.

``pydantic-settings`` is preferred, but if it is not installed the
settings fall back to a plain ``os.environ`` reader with the same
attributes/defaults, so the app still imports in a bare environment.
"""
from __future__ import annotations

import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # bare environment: fall back to os.environ below
    BaseSettings = object  # type: ignore[assignment,misc]
    SettingsConfigDict = None  # type: ignore[assignment]
    _HAS_PYDANTIC_SETTINGS = False


def _sanitize_no_proxy() -> None:
    """Strip bracketed IPv6 entries (``[::1]``) from NO_PROXY/no_proxy.

    httpx (used by qdrant-client and the Gemini REST client) raises
    ``InvalidURL: Invalid port`` while parsing bracketed entries in
    NO_PROXY. That made every Qdrant connect attempt fail and silently
    fall back to the in-memory store. Plain hostnames/IPs are kept.
    """
    for name in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(name)
        if not raw:
            continue
        kept = [e.strip() for e in raw.split(",")
                if e.strip() and not e.strip().startswith("[")]
        os.environ[name] = ",".join(kept)


_sanitize_no_proxy()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class _SettingsMixin:
    """Derived provider/dimension helpers, shared by both backends."""

    @property
    def embedding_dim(self) -> int:
        if getattr(self, "EMBEDDING_DIM", None):
            return int(self.EMBEDDING_DIM)
        if self.effective_embedding_provider == "gemini":
            return 768
        return 384

    @property
    def effective_embedding_provider(self) -> str:
        provider = (getattr(self, "EMBEDDING_PROVIDER", None) or "auto").lower()
        if provider == "auto":
            return "gemini" if getattr(self, "GEMINI_API_KEY", "") else "local"
        return provider

    @property
    def effective_llm_provider(self) -> str:
        """'gemini' | 'ollama' | 'none' — derived only from env vars."""
        if getattr(self, "GEMINI_API_KEY", ""):
            return "gemini"
        if getattr(self, "LLM_BASE_URL", ""):
            return "ollama"
        return "none"

    @property
    def web_search_enabled(self) -> bool:
        """Tavily fallback is on only when switched on AND a key is set.

        TAVILY_ENABLED defaults to true, so a key on its own turns the
        fallback on; with no key it is off regardless of the switch.
        """
        return bool(getattr(self, "TAVILY_ENABLED", True)) and bool(
            (getattr(self, "TAVILY_API_KEY", "") or "").strip())


if _HAS_PYDANTIC_SETTINGS:

    class Settings(_SettingsMixin, BaseSettings):  # type: ignore[valid-type,misc]
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                          extra="ignore")

        # Vector store (Qdrant)
        QDRANT_URL: str = "http://localhost:6333"
        QDRANT_API_KEY: str = ""
        QDRANT_COLLECTION: str = "adaptive_rag"
        # Connect/probe timeout (s): the 5s default was too short for a
        # cold TLS connection to Qdrant Cloud through the egress proxy.
        QDRANT_TIMEOUT: int = 30

        # LLM / embeddings (Gemini is the default provider)
        GEMINI_API_KEY: str = ""
        GEMINI_MODEL: str = "gemini-3.5-flash-lite"
        GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
        # "gemini" | "local" | "hash" | "auto"
        EMBEDDING_PROVIDER: str = "auto"
        # None -> provider default (768 Gemini, 384 local/hash)
        EMBEDDING_DIM: int | None = None
        EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

        # LLM fallback: Ollama-compatible endpoint
        LLM_BASE_URL: str = "http://localhost:11434/v1"
        LLM_API_KEY: str = ""
        LLM_MODEL: str = "llama3.1"
        OPENAI_API_KEY: str = ""

        # Database: SQLite default; Neon Postgres URL also works
        DATABASE_URL: str = "sqlite+aiosqlite:///./adaptive_rag.db"

        # Tracing / observability (both optional)
        LANGSMITH_API_KEY: str = ""
        LANGSMITH_PROJECT: str = "adaptive-rag"
        OTEL_EXPORTER_OTLP_ENDPOINT: str = ""

        # Retrieval behaviour / chunking
        TOP_K: int = 5
        MAX_RETRIES: int = 2
        CHUNK_SIZE: int = 1000
        CHUNK_OVERLAP: int = 200
        # LLM routing in analyze_query burns one generate call per
        # question; the heuristic already routes real questions to
        # retrieval. Only enable it (USE_LLM_ROUTING=true) if you want
        # the LLM to classify ambiguous questions — it costs quota.
        USE_LLM_ROUTING: bool = False

        # Conversational history + prompt token budget (see graph.py).
        # MAX_CONTEXT_TOKENS caps the assembled generation prompt
        # (system + history + retrieved context + question, chars/4
        # estimate): oldest history turns are dropped first, then the
        # lowest-graded chunks. MAX_OUTPUT_TOKENS caps each generation
        # call (previously uncapped). MAX_HISTORY_TURNS is how many
        # prior turns (user+assistant pairs) ride along with each
        # question for follow-up resolution.
        MAX_CONTEXT_TOKENS: int = 8000
        MAX_OUTPUT_TOKENS: int = 1024
        MAX_HISTORY_TURNS: int = 3

        # Guardrails (see guardrails.py). Blocked topics are category
        # names (compensation, personal_data, credentials) plus extra
        # literal phrases. PII types select the output scanner's entity
        # set (email, phone, pan, aadhaar, api_key, credit_card, person).
        # GUARDRAILS_ENABLED=false turns all three layers off for a
        # local demo; default is ON.
        GUARDRAILS_ENABLED: bool = True
        GUARDRAIL_DEFAULT_ROLE: str = "employee"
        GUARDRAIL_BLOCKED_TOPICS: str = "compensation,personal_data,credentials"
        GUARDRAIL_BLOCKED_PHRASES: str = ""
        GUARDRAIL_PII_TYPES: str = "email,phone,pan,aadhaar,api_key,credit_card"
        GUARDRAIL_STREAM_SCAN: bool = True
        # Roles that may ask about the compensation / personal_data
        # topics (still role-filtered at retrieval). Credentials are
        # blocked for every role, privileged or not.
        GUARDRAIL_PRIVILEGED_ROLES: str = "exec,hr"

        # Response cache (see cache.py): exact-match + semantic, Redis
        # with a bounded in-process fallback when Redis is unreachable.
        # CACHE_TTL_SECONDS: how long a cached answer lives (default 24h).
        # CACHE_SEMANTIC_THRESHOLD: cosine similarity for a paraphrase
        # to count as a cache hit. CACHE_MAX_ENTRIES bounds the
        # in-process fallback; CACHE_SEMANTIC_SCAN_MAX bounds how many
        # stored embeddings a semantic lookup scans.
        REDIS_URL: str = "redis://localhost:6379"
        CACHE_TTL_SECONDS: int = 86400
        CACHE_SEMANTIC_THRESHOLD: float = 0.95
        CACHE_MAX_ENTRIES: int = 200
        CACHE_SEMANTIC_SCAN_MAX: int = 500

        # Web search fallback (Tavily — see app/websearch.py). The
        # classic Adaptive RAG fallback: when local grading rejects
        # everything (or the grounding check fails), search the web and
        # answer from those results. TAVILY_ENABLED=false turns it off;
        # with no key it is off anyway (see web_search_enabled). Basic
        # depth costs 1 Tavily credit per search on the free Researcher
        # plan (1,000 credits/month); advanced costs 2.
        TAVILY_API_KEY: str = ""
        TAVILY_ENABLED: bool = True
        TAVILY_SEARCH_DEPTH: str = "basic"
        TAVILY_MAX_RESULTS: int = 5

else:

    class Settings(_SettingsMixin):  # type: ignore[no-redef]
        """Plain os.environ-backed settings (no pydantic-settings installed)."""

        def __init__(self) -> None:
            env = os.environ.get
            self.QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
            self.QDRANT_API_KEY = env("QDRANT_API_KEY", "")
            self.QDRANT_COLLECTION = env("QDRANT_COLLECTION", "adaptive_rag")
            self.QDRANT_TIMEOUT = _env_int("QDRANT_TIMEOUT", 30)
            self.GEMINI_API_KEY = env("GEMINI_API_KEY", "")
            self.GEMINI_MODEL = env("GEMINI_MODEL", "gemini-3.5-flash-lite")
            self.GEMINI_EMBEDDING_MODEL = env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
            self.EMBEDDING_PROVIDER = env("EMBEDDING_PROVIDER", "auto")
            self.EMBEDDING_DIM = _env_optional_int("EMBEDDING_DIM")
            self.EMBEDDING_MODEL = env("EMBEDDING_MODEL",
                                       "sentence-transformers/all-MiniLM-L6-v2")
            self.LLM_BASE_URL = env("LLM_BASE_URL", "http://localhost:11434/v1")
            self.LLM_API_KEY = env("LLM_API_KEY", "")
            self.LLM_MODEL = env("LLM_MODEL", "llama3.1")
            self.OPENAI_API_KEY = env("OPENAI_API_KEY", "")
            self.DATABASE_URL = env("DATABASE_URL", "sqlite+aiosqlite:///./adaptive_rag.db")
            self.LANGSMITH_API_KEY = env("LANGSMITH_API_KEY", "")
            self.LANGSMITH_PROJECT = env("LANGSMITH_PROJECT", "adaptive-rag")
            self.OTEL_EXPORTER_OTLP_ENDPOINT = env("OTEL_EXPORTER_OTLP_ENDPOINT", "")
            self.TOP_K = _env_int("TOP_K", 5)
            self.MAX_RETRIES = _env_int("MAX_RETRIES", 2)
            self.USE_LLM_ROUTING = os.environ.get("USE_LLM_ROUTING", "").lower() in ("1", "true", "yes")
            self.CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
            self.CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 200)
            self.MAX_CONTEXT_TOKENS = _env_int("MAX_CONTEXT_TOKENS", 8000)
            self.MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 1024)
            self.MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 3)
            self.GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "true").lower() in ("1", "true", "yes")
            self.GUARDRAIL_DEFAULT_ROLE = env("GUARDRAIL_DEFAULT_ROLE", "employee")
            self.GUARDRAIL_BLOCKED_TOPICS = env("GUARDRAIL_BLOCKED_TOPICS", "compensation,personal_data,credentials")
            self.GUARDRAIL_BLOCKED_PHRASES = env("GUARDRAIL_BLOCKED_PHRASES", "")
            self.GUARDRAIL_PII_TYPES = env("GUARDRAIL_PII_TYPES", "email,phone,pan,aadhaar,api_key,credit_card")
            self.GUARDRAIL_STREAM_SCAN = os.environ.get("GUARDRAIL_STREAM_SCAN", "true").lower() in ("1", "true", "yes")
            self.GUARDRAIL_PRIVILEGED_ROLES = env("GUARDRAIL_PRIVILEGED_ROLES", "exec,hr")
            self.REDIS_URL = env("REDIS_URL", "redis://localhost:6379")
            self.CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 86400)
            try:
                self.CACHE_SEMANTIC_THRESHOLD = float(
                    env("CACHE_SEMANTIC_THRESHOLD", "0.95"))
            except ValueError:
                self.CACHE_SEMANTIC_THRESHOLD = 0.95
            self.CACHE_MAX_ENTRIES = _env_int("CACHE_MAX_ENTRIES", 200)
            self.CACHE_SEMANTIC_SCAN_MAX = _env_int("CACHE_SEMANTIC_SCAN_MAX", 500)

            # Web search fallback (Tavily — see app/websearch.py).
            self.TAVILY_API_KEY = env("TAVILY_API_KEY", "")
            self.TAVILY_ENABLED = os.environ.get("TAVILY_ENABLED", "true").lower() in ("1", "true", "yes")
            self.TAVILY_SEARCH_DEPTH = env("TAVILY_SEARCH_DEPTH", "basic")
            self.TAVILY_MAX_RESULTS = _env_int("TAVILY_MAX_RESULTS", 5)


settings = Settings()

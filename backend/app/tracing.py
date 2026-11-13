"""Tracing: LangSmith (optional) + OpenTelemetry (optional).

Both are best-effort and never crash the app when unavailable:
no key / no endpoint / missing package all degrade to a no-op tracer.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable

from app.config import settings

logger = logging.getLogger(__name__)
_tracer = None
_langsmith_enabled = False


def setup_tracing() -> None:
    """Enable LangSmith and/or OTel from env vars. Safe to call always."""
    global _tracer, _langsmith_enabled
    if settings.LANGSMITH_API_KEY:
        try:
            os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
            os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
            os.environ.setdefault("LANGSMITH_TRACING", "true")
            from langsmith import Client  # noqa: F401  (validates the install)

            _langsmith_enabled = True
            logger.info("Tracing: LangSmith enabled (project=%s)", settings.LANGSMITH_PROJECT)
        except Exception as exc:
            logger.warning("Tracing: LangSmith unavailable (%s); continuing without it.", exc)

    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    if endpoint:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer("adaptive-rag")
            logger.info("Tracing: OpenTelemetry OTLP enabled (endpoint=%s)", endpoint)
        except Exception as exc:
            logger.warning("Tracing: OpenTelemetry unavailable (%s); continuing without it.", exc)
    else:
        logger.info("Tracing: no OTLP endpoint set — OpenTelemetry is a no-op.")


def get_tracer():
    return _tracer


def is_langsmith_enabled() -> bool:
    return _langsmith_enabled


def traced(name: str | None = None):
    """Decorator: wrap a function in an OTel span when tracing is on,
    otherwise call it unchanged."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            if _tracer is None:
                return fn(*args, **kwargs)
            with _tracer.start_as_current_span(name or fn.__name__):
                return fn(*args, **kwargs)
        return wrapper
    return decorator

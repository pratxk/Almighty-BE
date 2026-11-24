"""In-memory store for per-run traces.

A run trace is the full transparency record for one ``POST /api/chat``
call (corpus stats, per-node timings, retrieved/kept/discarded counts,
LLM token counts). Traces are kept here for ``GET /api/runs/{id}`` and
their summary is also persisted on the conversation row (see ``db.py``).

Per-user isolation: every trace is stamped with its owner's id at
save time. ``get_run_trace`` returns a trace only to its owner — a
trace saved before auth (no owner recorded) is never shown to an
authenticated user, and another user's trace looks missing (404).
"""
from __future__ import annotations

RUN_TRACES: dict[str, dict] = {}
RUN_TRACE_OWNERS: dict[str, str | None] = {}


def save_run_trace(trace: dict, owner_id: str | None = None) -> dict:
    run_id = trace.get("run_id")
    if run_id:
        RUN_TRACES[run_id] = trace
        RUN_TRACE_OWNERS[run_id] = owner_id
    return trace


def get_run_trace(run_id: str, owner_id: str | None = None) -> dict | None:
    trace = RUN_TRACES.get(run_id)
    if trace is None:
        return None
    owner = RUN_TRACE_OWNERS.get(run_id)
    if owner is not None and owner != owner_id:
        return None
    if owner is None and owner_id is not None:
        return None
    return trace

"""In-memory store for per-run traces.

A run trace is the full transparency record for one ``POST /api/chat``
call (corpus stats, per-node timings, retrieved/kept/discarded counts,
LLM token counts). Traces are kept here for ``GET /api/runs/{id}`` and
their summary is also persisted on the conversation row (see ``db.py``).
"""
from __future__ import annotations

RUN_TRACES: dict[str, dict] = {}


def save_run_trace(trace: dict) -> dict:
    run_id = trace.get("run_id")
    if run_id:
        RUN_TRACES[run_id] = trace
    return trace


def get_run_trace(run_id: str) -> dict | None:
    return RUN_TRACES.get(run_id)

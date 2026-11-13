#!/usr/bin/env python3
"""Re-apply the venv compatibility shims RAGAS 0.4.x needs.

Why: ragas 0.4.3 does
    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI
but langchain-community 0.4.x removed the VertexAI modules (they moved
to langchain-google-vertexai). These two tiny re-export modules restore
the old import paths without downgrading langchain-core (which the app
and langgraph need at >=1.4).

Run after any fresh `pip install -r requirements.txt` in backend/.venv:
    .venv/bin/python backend/scripts/apply_venv_shims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SHIMS = {
    "langchain_community/chat_models/vertexai.py": (
        '"""Compat shim: re-export ChatVertexAI (moved to langchain-google-vertexai)."""\n'
        "from langchain_google_vertexai import ChatVertexAI\n\n"
        '__all__ = ["ChatVertexAI"]\n'
    ),
    "langchain_community/llms/vertexai.py": (
        '"""Compat shim: re-export VertexAI (moved to langchain-google-vertexai)."""\n'
        "from langchain_google_vertexai import VertexAI, VertexAIModelGarden\n\n"
        '__all__ = ["VertexAI", "VertexAIModelGarden"]\n'
    ),
}


def main() -> int:
    site = Path(sys.executable).resolve().parent.parent / "lib" / \
        f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    # Fall back to sysconfig if the layout differs.
    if not (site / "langchain_community").exists():
        import sysconfig
        site = Path(sysconfig.get_paths()["purelib"])
    for rel, content in SHIMS.items():
        target = site / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Adaptive RAG workflow (LangGraph StateGraph, with a sequential fallback).

State: question, rewritten_question, documents, route, grade, answer,
confidence, retries, trace. Nodes:

  analyze_query -> route: "direct" | "retrieve" | "clarify"
  retrieve      -> hybrid search, dynamic top_k (3 simple / 5 / 8 complex)
  grade_documents -> cosine + keyword-overlap score, drop < 0.25
  rewrite_query -> deterministic rewriter (+ optional LLM), loops to retrieve
  generate      -> LLM-grounded answer, else labelled extractive fallback
  check_grounding -> drop unsupported sentences / lower confidence /
                     rewrite again while retries remain

Edges: analyze -> direct => generate, retrieve => retrieve, clarify => END.
If LangGraph is not installed, the same nodes run sequentially in
``_run_sequential`` — identical behaviour, no LangGraph dependency.
``run_adaptive_rag(question)`` returns {answer, confidence, sources,
trace, route, retries, run_id, ...}; the trace carries corpus stats,
retrieved/kept/discarded counts, the rewritten query, per-LLM-call
token counts (real usage when returned, chars/4 estimate otherwise)
and per-node timings.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from app.config import settings
from app.embeddings import cosine_similarity, embed_query, estimate_tokens
from app.guardrails import (REFUSAL_MESSAGE, check_input, check_output,
                            stream_scan)
from app.llm import get_llm_client, llm_available
from app.run_store import save_run_trace
from app.vectorstore import get_vector_store
from app.websearch import result_domains, search_web, web_search_available

logger = logging.getLogger(__name__)

GRADE_THRESHOLD = 0.25
_WORD_RE = re.compile(r"[a-z0-9]+")
_GREETINGS = {"hi", "hello", "hey", "hii", "hiii", "good morning", "good afternoon",
              "good evening", "yo", "namaste"}
_VAGUE = {"what?", "explain", "tell me more", "and?", "why?", "how?", "what about it?",
          "and then?", "elaborate", "more", "huh?"}
_SYNONYMS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "llms": "large language models",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "api": "application programming interface",
    "db": "database",
    "k8s": "kubernetes",
}
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
              "it", "its", "this", "that", "with", "for", "on", "as", "by", "at", "from",
              "what", "which", "how", "why", "when", "who", "does", "do", "did", "can",
              "is", "tell", "me", "about", "please", "explain"}


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _new_trace(question: str, run_id: str, corpus_stats: dict | None,
               role: str | None = None) -> dict:
    return {
        "run_id": run_id, "question": question,
        "corpus": corpus_stats or {"documents_ingested": 0, "total_chars": 0,
                                   "total_tokens_est": 0, "total_chunks": 0,
                                   "chunk_size": settings.CHUNK_SIZE,
                                   "chunk_overlap": settings.CHUNK_OVERLAP},
        "steps": [], "timings_ms": {}, "llm_calls": [],
        "total_llm_tokens_est": 0, "total_llm_tokens": None,
        "retrieved": 0, "kept": 0, "discarded": 0,
        "rewritten_question": None, "route": None, "retries": 0,
        "contextualized_question": None, "history_turns": 0,
        "context_budget": {},
        # Web-search fallback (Tavily — see websearch.py): filled in by
        # the web_search node when it runs; "used" stays False otherwise.
        "web_search": {"used": False, "results": 0, "domains": [],
                       "forced": False},
        # Guardrails (guardrails.py): input/output decisions for this run.
        "guardrail": {
            "input": "pass", "output": "pass",
            "role": role or settings.GUARDRAIL_DEFAULT_ROLE,
            "input_reason": None, "input_category": None,
            "privileged_bypass": None,
            "output_findings": [], "stream_cut": False,
        },
    }


def _finish_step(state: dict, node: str, started: float, **extra: Any) -> None:
    ms = round((time.perf_counter() - started) * 1000, 1)
    state["trace"]["steps"].append({"node": node, "duration_ms": ms, **extra})
    state["trace"]["timings_ms"][node] = ms


def _stream_collect(client, prompt: str, system: str | None,
                    on_token) -> tuple[str | None, dict | None]:
    """Consume a client's generate_stream; return (text, usage).

    Text is None when nothing streamed (caller falls back to the blocking
    call). A mid-stream failure keeps whatever deltas already arrived and
    returns them with estimated usage instead of raising.
    """
    parts: list[str] = []
    usage: dict | None = None
    try:
        for event in client.generate_stream(prompt, system=system):
            if event.get("delta"):
                parts.append(event["delta"])
                on_token(event["delta"])
            if event.get("done"):
                usage = event.get("usage")
    except Exception as exc:
        if not parts:
            return None, None
        logger.warning("LLM stream broke mid-answer (%s); keeping %d chars "
                       "of partial text.", client.provider, sum(map(len, parts)))
    text = "".join(parts).strip()
    if not text:
        return None, None
    return text, usage


def _llm_generate(state: dict, node: str, prompt: str, system: str | None = None,
                  on_token=None) -> str | None:
    """One LLM call, recorded in the trace with token counts. None if no LLM/failure.

    When ``on_token`` is given (streaming runs), the call streams and each
    text delta is passed to the callback as it arrives; if streaming fails
    before any delta, it falls back to the blocking ``generate`` call.
    """
    client = get_llm_client()
    call: dict[str, Any] = {"node": node, "provider": client.provider,
                            "model": getattr(client, "model", ""),
                            "prompt_chars": len(prompt),
                            "prompt_tokens_est": estimate_tokens(prompt)}
    text: str | None = None
    usage: dict[str, Any] = {}
    if on_token is not None and hasattr(client, "generate_stream"):
        text, stream_usage = _stream_collect(client, prompt, system, on_token)
        usage = stream_usage or {}
    if text is None:
        try:
            result = client.generate(prompt, system=system)
        except Exception as exc:
            call["error"] = str(exc)
            call["completion_tokens_est"] = 0
            call["total_tokens_est"] = call["prompt_tokens_est"]
            state["trace"]["llm_calls"].append(call)
            state["trace"]["total_llm_tokens_est"] += call["total_tokens_est"]
            logger.warning("LLM call failed in %s (%s): %s", node, client.provider, exc)
            return None
        usage = result.get("usage") or {}
        text = result.get("text") or None
    usage = usage or {}
    call.update({
        "completion_tokens_est": usage.get("completion_tokens_est", 0),
        "total_tokens_est": usage.get("total_tokens_est", 0),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })
    state["trace"]["llm_calls"].append(call)
    state["trace"]["total_llm_tokens_est"] += call["total_tokens_est"]
    if call.get("total_tokens") is not None:
        prev = state["trace"]["total_llm_tokens"] or 0
        state["trace"]["total_llm_tokens"] = prev + call["total_tokens"]
    return text


# ---------------------------------------------------------------- history

# Words signalling a question refers back to the conversation ("that",
# "the second one", "tell me more") rather than standing on its own.
_ANAPHORA_WORDS = {
    "it", "its", "this", "that", "these", "those", "them", "they", "he", "she",
    "his", "her", "their", "same", "other", "more", "second", "first", "third",
    "fourth", "fifth", "above", "previous", "earlier", "also", "else", "another",
    "both", "either", "former", "latter",
}


def _normalize_history(history) -> list[dict]:
    """Sanitize client-sent history: only user/assistant turns, content
    capped per message, most recent MAX_HISTORY_TURNS pairs kept."""
    if not isinstance(history, list):
        return []
    cleaned: list[dict] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    return cleaned[-(settings.MAX_HISTORY_TURNS * 2):]


def _history_text(history: list[dict]) -> str:
    lines = []
    for msg in history or []:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _history_turns(history: list[dict]) -> list[list[dict]]:
    """Group messages into turns: a user message plus the assistant reply
    that follows it; a trailing lone user message is its own turn. Turns
    are the unit the token budget drops, oldest first."""
    turns: list[list[dict]] = []
    for msg in history or []:
        if msg.get("role") == "user" or not turns or turns[-1][0].get("role") != "user":
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def _needs_contextualization(question: str) -> bool:
    """Cheap gate: only spend an LLM rewrite call when the question looks
    like a follow-up — anaphoric words, ordinals, or a very short question.
    A full standalone question skips the call entirely."""
    words = set(_WORD_RE.findall((question or "").lower()))
    if words & _ANAPHORA_WORDS:
        return True
    return len(words) <= 4


def contextualize_query(state: dict) -> dict:
    """Pre-step (runs before routing/retrieval in both runners, not a
    LangGraph node — the graph cannot run before its entry point, and the
    sequential path shares this code): resolve follow-up references
    against recent history into a standalone question.

    Production pattern is contextualize-then-retrieve: retrieval runs on
    the standalone question, so "tell me more about that" / "what about
    the second one" actually resolves instead of retrieving on pronouns.
    With no history, a greeting, or an already-standalone question this
    is a no-op trace entry and the run behaves exactly as before. If the
    LLM rewrite fails or no LLM is configured, the raw question is used
    and history still reaches the generation prompt.
    """
    started = time.perf_counter()
    question = (state.get("question") or "").strip()
    history = state.get("history") or []
    standalone = question
    lowered = question.lower().strip(" .!?")
    if not history:
        method = "skipped_no_history"
    elif lowered in _GREETINGS:
        method = "skipped_greeting"
    elif not _needs_contextualization(question):
        method = "skipped_standalone"
    elif llm_available():
        text = _llm_generate(
            state, "contextualize_query",
            "Given the conversation below, rewrite the latest user question "
            "as a standalone question that can be understood without the "
            "conversation. If it is already standalone, return it unchanged. "
            "Reply with the question only.\n\n"
            + _history_text(history)
            + f"\nLatest question: {question}",
            system="You rewrite follow-up questions as standalone questions. "
                   "Reply with the question only.")
        if text:
            standalone = text.strip().strip("\"").strip()
            method = "llm"
        else:
            method = "llm_failed_raw"
    else:
        method = "no_llm_raw"
    _finish_step(state, "contextualize_query", started, method=method,
                 standalone_question=standalone, history_messages=len(history))
    state["trace"]["contextualized_question"] = (
        standalone if standalone != question else None)
    state["trace"]["history_turns"] = len(_history_turns(history))
    if standalone != question:
        state["rewritten_question"] = standalone
    return {"rewritten_question": state.get("rewritten_question", "")}


def _fit_prompt_context(docs: list[dict], history: list[dict],
                        fixed_tokens: int) -> tuple[list[dict], list[dict], dict]:
    """Enforce MAX_CONTEXT_TOKENS on the assembled generation prompt.

    Drops the oldest history turns first, then the lowest-graded retrieved
    chunks (always keeping at least one), until the prompt fits. Returns
    (docs_used, history_used, budget_info) — the info lands in the trace
    and the transparency panel, so truncation is visible, never silent.
    """
    budget = settings.MAX_CONTEXT_TOKENS
    turns = _history_turns(history)
    # Drop order for docs: lowest grade first; survivors keep retrieval order.
    drop_order = sorted(range(len(docs)),
                        key=lambda i: float(docs[i].get("grade", docs[i].get("score", 0.0))))
    kept_idx = set(range(len(docs)))

    def total_tokens() -> int:
        hist_tokens = sum(estimate_tokens(m.get("content", ""))
                          for t in turns for m in t)
        doc_tokens = sum(estimate_tokens(docs[i].get("text", ""))
                         for i in kept_idx)
        return fixed_tokens + hist_tokens + doc_tokens

    dropped_turns = 0
    while turns and total_tokens() > budget:
        turns.pop(0)
        dropped_turns += 1
    dropped_docs = 0
    while len(kept_idx) > 1 and total_tokens() > budget:
        kept_idx.discard(drop_order.pop(0))
        dropped_docs += 1
    docs_used = [d for i, d in enumerate(docs) if i in kept_idx]
    history_used = [m for t in turns for m in t]
    info = {
        "max_context_tokens": budget,
        "prompt_tokens_est": total_tokens(),
        "history_turns_included": len(turns),
        "history_turns_dropped": dropped_turns,
        "docs_dropped": dropped_docs,
        "truncated": dropped_turns > 0 or dropped_docs > 0,
    }
    return docs_used, history_used, info


# ------------------------------------------------------------- guardrails

def _apply_input_guardrail(state: dict, question: str, role: str) -> dict:
    """Input guardrail for one question text. Records a trace step
    (pass or block) and returns the verdict dict from guardrails.py."""
    started = time.perf_counter()
    verdict = check_input(question, role)
    _finish_step(state, "guardrails_input", started,
                 status="block" if not verdict["allowed"] else "pass",
                 category=verdict.get("category"),
                 reason=verdict.get("reason"),
                 privileged_bypass=verdict.get("privileged_bypass"))
    state["trace"]["guardrail"]["privileged_bypass"] = verdict.get(
        "privileged_bypass")
    if not verdict["allowed"]:
        state["trace"]["guardrail"]["input"] = "block"
        state["trace"]["guardrail"]["input_reason"] = verdict.get("reason")
        state["trace"]["guardrail"]["input_category"] = verdict.get("category")
    return verdict


def _refuse(state: dict) -> None:
    """Turn the run into a generic refusal: no answer, no sources, no
    retrieval results leak. The refusal message reveals nothing about
    what was detected or what exists in the documents."""
    state["answer"] = REFUSAL_MESSAGE
    state["confidence"] = 0.0
    state["route"] = "blocked"
    state["documents"] = []


def _apply_output_guardrail(state: dict) -> None:
    """Output guardrail on the finished answer (skipped for clarify /
    blocked runs, whose answers are canned). Redacts PII spans in place;
    refuses the answer outright when it is basically PII."""
    answer = state.get("answer") or ""
    if state.get("route") in ("clarify", "blocked") or not answer.strip():
        return
    started = time.perf_counter()
    verdict = check_output(answer, state.get("role"))
    state["trace"]["guardrail"]["output"] = verdict["status"]
    state["trace"]["guardrail"]["output_findings"] = verdict["findings"]
    _finish_step(state, "guardrails_output", started,
                 status=verdict["status"],
                 findings=[{"type": f["type"]} for f in verdict["findings"]])
    if verdict["status"] == "block":
        state["answer"] = REFUSAL_MESSAGE
        state["confidence"] = 0.0
        state["documents"] = []
    elif verdict["status"] == "redact":
        state["answer"] = verdict["answer"]


# ---------------------------------------------------------------- nodes

def analyze_query(state: dict) -> dict:
    started = time.perf_counter()
    # Route on the contextualized (standalone) question when one exists —
    # a raw "tell me more about that" would otherwise hit the vague
    # branch and ask for clarification instead of retrieving.
    question = ((state.get("rewritten_question") or state.get("question") or "")
                .strip())
    lowered = question.lower().strip(" .!?")
    route = "retrieve"
    method = "heuristic"
    if lowered in _GREETINGS or (len(question.split()) <= 3 and lowered.split(" ")[0] in _GREETINGS):
        route = "direct"
    elif lowered in _VAGUE or (len(question.split()) <= 2 and "?" in question):
        route = "clarify"
    elif llm_available() and getattr(settings, "USE_LLM_ROUTING", False):
        # LLM routing only when explicitly enabled (USE_LLM_ROUTING=true):
        # it costs one generate call per question against a tight free-tier
        # quota, and the heuristic above already routes real questions to
        # retrieval. Any failure falls back to the heuristic.
        text = _llm_generate(
            state, "analyze_query",
            "Classify this user question with exactly one word — "
            "direct (greeting/smalltalk, no retrieval needed), retrieve "
            "(needs documents), or clarify (too vague/ambiguous to answer).\n"
            f"Question: {question}",
            system="You are a query router. Reply with one word only.")
        if text:
            word = _WORD_RE.findall(text.lower())
            if word and word[0] in ("direct", "retrieve", "clarify"):
                route = word[0]
                method = "llm"
    _finish_step(state, "analyze_query", started, route=route, method=method)
    state["trace"]["route"] = route
    return {"route": route}


def retrieve(state: dict) -> dict:
    started = time.perf_counter()
    query = state.get("rewritten_question") or state.get("question") or ""
    words = len(query.split())
    complex_markers = (" and ", "compare", "difference", "versus", " vs ", "differences")
    is_complex = words > 12 or any(m in query.lower() for m in complex_markers)
    top_k = 8 if is_complex else (3 if words <= 5 else settings.TOP_K)
    # Retrieval must never 500 the chat: if embeddings or the vector
    # store are down (e.g. Gemini embedding quota exhausted), continue
    # with no local docs — grading keeps 0, so the graph falls through
    # to the web-search fallback or a clean decline.
    try:
        qvec = embed_query(query)
        docs = get_vector_store().search(query, qvec, top_k=top_k,
                                         role=state.get("role"),
                                         owner_id=state.get("owner_id"))
    except Exception as exc:  # noqa: BLE001 - retrieval must not break chat
        logger.warning("retrieve: local search unavailable (%s); continuing "
                       "with no local docs.", exc)
        docs = []
    # Guardrails: the role filter runs inside search (Qdrant-side filter
    # + pool backstop), so restricted chunks never reach grading.
    _finish_step(state, "retrieve", started, query=query, top_k=top_k,
                 retrieved=len(docs), complexity="complex" if is_complex else
                 ("simple" if words <= 5 else "default"),
                 role=state.get("role"))
    state["trace"]["retrieved"] = len(docs)
    return {"documents": docs, "top_k": top_k}


def grade_documents(state: dict) -> dict:
    started = time.perf_counter()
    query = state.get("rewritten_question") or state.get("question") or ""
    qwords = _keywords(query)
    graded: list[dict] = []
    for doc in state.get("documents") or []:
        dwords = _keywords(doc.get("text", ""))
        overlap = (len(qwords & dwords) / len(qwords)) if qwords else 0.0
        grade = 0.6 * float(doc.get("score", 0.0)) + 0.4 * overlap
        graded.append({**doc, "grade": round(grade, 3), "keyword_overlap": round(overlap, 3)})
    kept = [d for d in graded if d["grade"] >= GRADE_THRESHOLD]
    discarded = len(graded) - len(kept)
    _finish_step(state, "grade_documents", started, graded=len(graded),
                 kept=len(kept), discarded=discarded, threshold=GRADE_THRESHOLD,
                 scores=[{"title": d.get("title", ""), "grade": d["grade"]} for d in graded])
    state["trace"]["kept"] = len(kept)
    state["trace"]["discarded"] = discarded
    return {"documents": kept, "all_graded": graded,
            "grade": {"kept": len(kept), "discarded": discarded, "threshold": GRADE_THRESHOLD}}


# ------------------------------------------------------- web search fallback

def _web_search_enabled() -> bool:
    """Tavily fallback is on only when enabled AND a key is configured."""
    return bool(web_search_available())


def _web_fallback_wanted(state: dict) -> bool:
    """Should the web-search node run now (once per run)?

    Yes when local retrieval failed — the grader kept nothing after all
    retries — or when the client forced web search for this question.
    """
    if not _web_search_enabled() or state.get("web_searched"):
        return False
    if state.get("force_web_search"):
        return True
    return (not state.get("documents")
            and int(state.get("retries") or 0) >= settings.MAX_RETRIES)


def web_search(state: dict) -> dict:
    """Web-search fallback node (Tavily).

    Appends web chunks to the documents list; the grader then scores
    them like local chunks and the generator answers from the kept ones,
    with web citations. Runs at most once per run (``web_searched``).
    Never raises: a failed search is an empty result with a trace step.
    """
    started = time.perf_counter()
    query = state.get("rewritten_question") or state.get("question") or ""
    results = search_web(query)
    domains = result_domains(results)
    _finish_step(state, "web_search", started, results=len(results),
                 domains=domains, forced=bool(state.get("force_web_search")),
                 status="ok" if results else "empty")
    state["trace"]["web_search"] = {
        "used": True, "results": len(results), "domains": domains,
        "forced": bool(state.get("force_web_search")),
    }
    existing = state.get("documents") or []
    return {"documents": existing + results, "web_searched": True}


def rewrite_query(state: dict) -> dict:
    started = time.perf_counter()
    # Rewrite the contextualized question when one exists, so a retrieval
    # retry keeps the resolved reference instead of the raw pronouns.
    question = (state.get("rewritten_question") or state.get("question") or "")
    retries = int(state.get("retries") or 0) + 1
    rewritten = None
    if llm_available():
        rewritten = _llm_generate(
            state, "rewrite_query",
            f"Rewrite this question as a concise search query, expanding "
            f"abbreviations and adding likely synonyms. Reply with the query only.\n"
            f"Question: {question}",
            system="You rewrite questions as search queries. Reply with the query only.")
    if not rewritten:
        words = [w for w in _WORD_RE.findall(question.lower()) if w not in _STOPWORDS]
        expanded: list[str] = []
        for w in words:
            expanded.append(w)
            if w in _SYNONYMS and _SYNONYMS[w] not in expanded:
                expanded.append(_SYNONYMS[w])
        rewritten = " ".join(dict.fromkeys(expanded)) or question
    _finish_step(state, "rewrite_query", started,
                 rewritten_question=rewritten, retries=retries, method="llm" if llm_available() else "deterministic")
    state["trace"]["rewritten_question"] = rewritten
    state["trace"]["retries"] = retries
    return {"rewritten_question": rewritten, "retries": retries}


def _extractive_answer(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs[:3], start=1):
        snippet = " ".join((doc.get("text") or "").split())[:600]
        parts.append(f"[{i}] {snippet}")
    body = "\n\n".join(parts)
    return ("(Extractive summary — no LLM is configured, so this answer is "
            "assembled directly from the retrieved passages below.)\n\n" + body)


def generate(state: dict, on_token=None) -> dict:
    """Generate node. ``on_token`` (streaming runs only) receives each LLM
    text delta as it arrives; the blocking path passes nothing.

    The generation prompt carries (a) the recent conversation turns, so
    answers can reference what was already said, and (b) the retrieved
    document context — fitted to MAX_CONTEXT_TOKENS first (oldest history
    turns, then lowest-graded chunks), with the fit recorded in the trace.
    History is only for resolving references; every factual claim must
    still come from the retrieved context, which the grounding check
    enforces against the documents alone.
    """
    started = time.perf_counter()
    route = state.get("route")
    docs = state.get("documents") or []
    if not docs and state.get("all_graded"):
        # Retries exhausted with nothing passing grading: answer from the
        # best-graded passages rather than nothing, flagged by low confidence.
        docs = sorted(state["all_graded"], key=lambda d: -d.get("grade", 0))[:3]
    history = state.get("history") or []
    # Ask about the standalone question when one exists; the trace keeps
    # the user's raw question.
    question = (state.get("rewritten_question") or state.get("question") or "")
    answer, confidence, llm_used = "", 0.0, False

    if route == "direct" and not docs:
        lowered = question.lower().strip(" .!?")
        if lowered in _GREETINGS:
            answer = "Hello! Ask me anything about the documents you have ingested."
            confidence = 1.0
        elif llm_available():
            text = _llm_generate(state, "generate",
                                 f"Reply briefly and helpfully to: {question}",
                                 on_token=on_token)
            if text:
                answer, confidence, llm_used = text, 0.7, True
        if not answer:
            answer = ("I don't know — I have no documents to answer from and no "
                      "LLM is configured for open-ended chat. Ingest some documents "
                      "and ask about them.")
            confidence = 0.0
    elif not docs:
        answer = ("I couldn't find relevant information in the ingested documents "
                  "to answer that. Try rephrasing, or ingest documents on the topic.")
        confidence = 0.0
    else:
        confidence = round(sum(float(d.get("grade", d.get("score", 0.0))) for d in docs) / len(docs), 3)
        if llm_available():
            system = "You answer strictly from the provided context, with citations."
            instruction = (
                "Answer the question using ONLY the retrieved context below. "
                "Cite sources as [1], [2], etc. If the context does not contain "
                "the answer, say so — do not invent anything. The conversation "
                "history is only to help you understand references in the "
                "question (like 'that' or 'the second one'); every factual "
                "claim must come from the retrieved context. "
                "When the user asks for a summary or explanation of several "
                "topics, give EACH topic a proper detailed paragraph — what it "
                "is, why it matters, how it works — instead of a one-line "
                "bullet. Use headings and lists to structure the answer, but "
                "explain every point; never just list topic titles.")
            # Fit history + docs to the token budget before assembling.
            fixed_tokens = estimate_tokens(system + instruction + question)
            docs, history_used, budget = _fit_prompt_context(
                docs, history, fixed_tokens)
            state["trace"]["context_budget"] = budget
            state["trace"]["history_turns"] = budget["history_turns_included"]
            history_text = _history_text(history_used)
            context = "\n\n".join(
                f"[{i}] ({d.get('title', '')} — {d.get('source', '')})\n{d.get('text', '')}"
                for i, d in enumerate(docs, start=1))
            prompt = (
                f"{instruction}\n\n"
                + (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
                + f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
            text = _llm_generate(state, "generate", prompt,
                                 system=system,
                                 on_token=on_token)
            if text:
                answer, llm_used = text, True
        if not answer:
            answer = _extractive_answer(docs)
    _finish_step(state, "generate", started, llm_used=llm_used,
                 docs_used=len(docs), confidence=confidence)
    return {"answer": answer, "confidence": confidence}


def check_grounding(state: dict) -> dict:
    started = time.perf_counter()
    answer = state.get("answer") or ""
    docs = state.get("documents") or state.get("all_graded") or []
    source_words: set[str] = set()
    for d in docs:
        source_words |= _keywords(d.get("text", ""))
    sentences = _sentences(answer)
    # The extractive label / disclaimer lines are not claims; skip them.
    claim_lines = [s for s in sentences
                   if not s.startswith("(Extractive summary") and "I don't know" not in s
                   and "couldn't find relevant" not in s]
    kept_sentences, dropped = [], 0
    for s in sentences:
        words = _keywords(s)
        if not words or s not in claim_lines or (words & source_words):
            kept_sentences.append(s)
        else:
            dropped += 1
    action = "ok"
    confidence = float(state.get("confidence") or 0.0)
    retries = int(state.get("retries") or 0)
    if claim_lines and dropped == len(claim_lines):
        if retries < settings.MAX_RETRIES:
            action = "rewrite"
        else:
            action = "disclaimer"
            confidence = min(confidence, 0.2)
            answer = (answer + "\n\nNote: I could not verify this answer against "
                      "the retrieved documents, so treat it with caution.")
    elif dropped:
        answer = " ".join(kept_sentences)
        action = "dropped_unsupported"
        confidence = round(confidence * 0.8, 3)
    _finish_step(state, "check_grounding", started, action=action,
                 sentences=len(sentences), dropped=dropped, confidence=confidence)
    return {"answer": answer, "confidence": confidence, "grounding_action": action}


# ---------------------------------------------------------------- graph

try:
    from langgraph.graph import END, StateGraph
    from typing import TypedDict

    class RAGState(TypedDict, total=False):
        question: str
        rewritten_question: str
        history: list
        role: str
        owner_id: str
        documents: list
        all_graded: list
        route: str
        grade: dict
        answer: str
        confidence: float
        retries: int
        trace: dict
        top_k: int
        grounding_action: str
        web_searched: bool
        force_web_search: bool

    def _after_grade(s: dict) -> str:
        """grade_documents -> generate | rewrite_query | web_search.

        Forced web search runs once before generating, even when local
        docs were kept. Otherwise local docs go straight to generate;
        with nothing kept, retry locally while retries remain, then fall
        back to the web once; with the web already tried, generate from
        whatever the grader kept (or the best-graded fallback passages).
        """
        if (s.get("force_web_search") and not s.get("web_searched")
                and _web_search_enabled()):
            return "web_search"
        if s.get("documents"):
            return "generate"
        if int(s.get("retries") or 0) < settings.MAX_RETRIES:
            return "rewrite_query"
        if _web_fallback_wanted(s):
            return "web_search"
        return "generate"

    def _after_grounding(s: dict) -> str:
        """check_grounding -> rewrite_query | web_search | END.

        A failed grounding check retries locally while retries remain
        (the node only emits "rewrite" while that is true); once local
        retries are spent, one web-search pass gets a chance before the
        run ends.
        """
        if s.get("grounding_action") != "rewrite":
            return END
        if int(s.get("retries") or 0) < settings.MAX_RETRIES:
            return "rewrite_query"
        if _web_search_enabled() and not s.get("web_searched"):
            return "web_search"
        return END

    def _build_graph():
        graph = StateGraph(RAGState)
        graph.add_node("analyze_query", analyze_query)
        graph.add_node("retrieve", retrieve)
        graph.add_node("grade_documents", grade_documents)
        graph.add_node("rewrite_query", rewrite_query)
        graph.add_node("web_search", web_search)
        graph.add_node("generate", generate)
        graph.add_node("check_grounding", check_grounding)
        graph.set_entry_point("analyze_query")
        graph.add_conditional_edges(
            "analyze_query",
            lambda s: s.get("route", "retrieve"),
            {"direct": "generate", "retrieve": "retrieve", "clarify": END})
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents", _after_grade,
            {"generate": "generate", "rewrite_query": "rewrite_query",
             "web_search": "web_search"})
        graph.add_edge("web_search", "grade_documents")
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", "check_grounding")
        graph.add_conditional_edges(
            "check_grounding", _after_grounding,
            {"rewrite_query": "rewrite_query", "web_search": "web_search",
             END: END})
        return graph.compile()

    _graph = _build_graph()
    logger.info("LangGraph adaptive-RAG graph compiled.")
except Exception as exc:  # langgraph not installed
    _graph = None
    logger.warning("LangGraph unavailable (%s); using sequential fallback.", exc)


def _run_sequential(state: dict) -> dict:
    """Same node order as the graph, for environments without LangGraph."""
    state.update(analyze_query(state))
    if state.get("route") == "clarify":
        state["answer"] = ("Could you clarify your question? Tell me what topic "
                           "or document you mean, or rephrase it as a full question.")
        state["confidence"] = 0.0
        return state
    if state.get("route") == "retrieve":
        for _ in range(settings.MAX_RETRIES + 1):
            state.update(retrieve(state))
            state.update(grade_documents(state))
            if state.get("documents") or int(state.get("retries") or 0) >= settings.MAX_RETRIES:
                break
            state.update(rewrite_query(state))
        # Web fallback: the grader kept nothing after all local retries
        # (or the client forced web search) — search the web once, then
        # grade those chunks like local ones.
        if _web_fallback_wanted(state):
            state.update(web_search(state))
            state.update(grade_documents(state))
    state.update(generate(state))
    state.update(check_grounding(state))
    if state.get("grounding_action") == "rewrite" and int(state.get("retries") or 0) < settings.MAX_RETRIES:
        state.update(rewrite_query(state))
        state.update(retrieve(state))
        state.update(grade_documents(state))
        state.update(generate(state))
        state.update(check_grounding(state))
    # Grounding still failing after local retries: one web-search pass,
    # then answer from the web chunks (or the best local passages).
    if (state.get("grounding_action") == "rewrite"
            and _web_search_enabled() and not state.get("web_searched")):
        state.update(web_search(state))
        state.update(grade_documents(state))
        state.update(generate(state))
        state.update(check_grounding(state))
    return state


def run_adaptive_rag(question: str, corpus_stats: dict | None = None,
                     history=None, role: str = "employee",
                     web_search_flag: bool = False,
                     owner_id: str | None = None) -> dict:
    """Run the adaptive RAG workflow. Returns the full result + trace.

    ``history`` is an optional list of prior turns ({role, content}) sent
    by the client; it is normalized (most recent MAX_HISTORY_TURNS pairs)
    and used to contextualize follow-ups and to ground the generation
    prompt. With no history the run behaves exactly as before.

    ``role`` is the requesting user's role (default "employee"). It
    drives the guardrails: the input check runs before any retrieval,
    retrieval itself is role-filtered, and the output check runs on the
    finished answer (see guardrails.py).

    ``owner_id`` is the requesting user's id. Retrieval is filtered to
    that user's documents only (see vectorstore.search) — a run without
    an owner retrieves nothing local. Web-search results are public
    web pages, so they are merged as before.

    ``web_search_flag`` forces the Tavily web-search fallback for this
    question even when local documents answer (the web chunks are
    merged with the local ones and graded together).
    """
    role = (role or "").strip().lower() or settings.GUARDRAIL_DEFAULT_ROLE
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    trace = _new_trace(question, run_id, corpus_stats, role=role)
    state: dict[str, Any] = {
        "question": question, "rewritten_question": "",
        "history": _normalize_history(history),
        "role": role, "owner_id": owner_id,
        "documents": [],
        "all_graded": [], "route": "retrieve", "grade": {}, "answer": "",
        "confidence": 0.0, "retries": 0, "trace": trace,
        "top_k": settings.TOP_K, "grounding_action": "ok",
        "web_searched": False, "force_web_search": bool(web_search_flag),
    }
    # Guardrail 1 — input check, before any retrieval or LLM spend.
    # The contextualized (standalone) question is checked too: a benign
    # follow-up ("what about that figure?") that resolves to a blocked
    # topic is still a blocked question.
    verdict = _apply_input_guardrail(state, question, role)
    blocked = not verdict["allowed"]
    if not blocked:
        state.update(contextualize_query(state))
        standalone = (state.get("rewritten_question") or "").strip()
        if standalone and standalone.lower() != question.strip().lower():
            verdict = _apply_input_guardrail(state, standalone, role)
            blocked = not verdict["allowed"]
    if blocked:
        _refuse(state)
    elif _graph is not None:
        try:
            state = dict(_graph.invoke(state))
            state.setdefault("trace", trace)
        except Exception as exc:
            logger.warning("LangGraph invoke failed (%s); falling back to sequential run.", exc)
            state = _run_sequential(state)
    else:
        state = _run_sequential(state)
    if not blocked:
        # Guardrail 2 — output check on the finished answer.
        _apply_output_guardrail(state)

    total_ms = round((time.perf_counter() - started) * 1000, 1)
    trace["total_duration_ms"] = total_ms
    trace["route"] = state.get("route")
    trace["retries"] = int(state.get("retries") or 0)
    trace["timings_ms"]["total"] = total_ms
    if state.get("route") == "clarify" and not state.get("answer"):
        state["answer"] = ("Could you clarify your question? Tell me what topic "
                           "or document you mean, or rephrase it as a full question.")
        state["confidence"] = 0.0
    docs = state.get("documents") or []
    sources = [{
        "title": d.get("title", ""), "source": d.get("source", ""),
        "score": round(float(d.get("grade", d.get("score", 0.0))), 3),
        "source_type": d.get("source_type", "local"),
        # 800 chars (was 300): a useful citation preview for the UI, and
        # long enough for the evaluator to measure retrieval honestly —
        # with 300-char snippets, keywords present in the chunk were
        # missed and hit rate was under-measured.
        "snippet": " ".join((d.get("text") or "").split())[:800],
    } for d in docs]
    result = {
        "run_id": run_id, "answer": state.get("answer", ""),
        "confidence": float(state.get("confidence") or 0.0),
        "sources": sources, "trace": trace, "route": state.get("route"),
        "retries": int(state.get("retries") or 0),
    }
    save_run_trace(trace, owner_id=owner_id)
    return result


# ---------------------------------------------------------------- streaming

def _step_detail(node: str, s: dict) -> str:
    """One-line human summary of a finished trace step (mirrors the
    frontend's trace.ts so live step events read the same as the final trace)."""
    if node == "analyze_query":
        return f"Route: {s.get('route', '?')} ({s.get('method', 'heuristic')})"
    if node == "contextualize_query":
        q = s.get("standalone_question", "")
        if q:
            return (f"Contextualized follow-up ({s.get('method', '?')}): {q}")
        return f"Contextualized the question ({s.get('method', '?')})"
    if node == "retrieve":
        return (f"Retrieved {s.get('retrieved', '?')} chunks "
                f"(top_k {s.get('top_k', '?')}, {s.get('complexity', '?')}) "
                f"in {s.get('duration_ms', '?')} ms")
    if node == "grade_documents":
        return (f"Graded {s.get('graded', '?')} chunks — kept {s.get('kept', '?')}, "
                f"discarded {s.get('discarded', '?')} (threshold {s.get('threshold', '?')})")
    if node == "rewrite_query":
        return (f"Rewrote query to: {s['rewritten_question']}"
                if s.get("rewritten_query") else "Rewrote the query and re-ran retrieval")
    if node == "generate":
        suffix = "" if s.get("llm_used") else " — extractive fallback, no LLM"
        return (f"Generated the answer from {s.get('docs_used', '?')} documents "
                f"(confidence {s.get('confidence', '?')}){suffix}")
    if node == "check_grounding":
        return (f"Grounding check: {s.get('action', '?')} — {s.get('dropped', 0)} "
                f"unsupported sentence(s) of {s.get('sentences', '?')}")
    if node == "guardrails_input":
        detail = f"Input guardrail: {s.get('status', '?')}"
        if s.get("status") == "block":
            detail += f" — blocked ({s.get('category', '?')})"
        elif s.get("privileged_bypass"):
            detail += f" — privileged bypass ({s.get('privileged_bypass')})"
        return detail
    if node == "guardrails_output":
        findings = s.get("findings") or []
        return (f"Output guardrail: {s.get('status', '?')}"
                + (f" — {len(findings)} finding(s) redacted"
                   if s.get("status") == "redact" else "")
                + (f" — answer refused"
                   if s.get("status") == "block" else ""))
    if node == "web_search":
        domains = s.get("domains") or []
        detail = f"Web search (Tavily): {s.get('results', '?')} result(s)"
        if domains:
            detail += f" from {', '.join(str(d) for d in domains)}"
        if s.get("forced"):
            detail += " (forced)"
        if s.get("status") == "empty":
            detail += " — no usable results"
        return detail
    return f"{node} completed in {s.get('duration_ms', '?')} ms"


def run_adaptive_rag_stream(question: str, corpus_stats: dict | None,
                            emit, history=None, role: str = "employee",
                            web_search_flag: bool = False,
                            owner_id: str | None = None) -> None:
    """Streaming twin of :func:`run_adaptive_rag`.

    Runs the same nodes in the same order (sequential path only — the
    LangGraph invoke API cannot surface per-node events), calling
    ``emit(event)`` as work happens:

    - ``{"type": "step", "node": ..., "status": "running" | "done",
      "detail": ..., "duration_ms": ...}`` around every node (plus a
      synthetic ``routing`` step after analysis, so the graph lights up);
    - ``{"type": "token", "delta": ...}`` for each LLM text delta while
      the answer generates (extractive fallback emits the whole answer
      as one ``token`` event so the frontend path stays identical);
    - ``{"type": "done", <full result>}`` at the end — same payload shape
      as the blocking endpoint, so the frontend can swap in final numbers.

    ``history`` is the same optional prior-turns list as the blocking
    twin; the contextualize step emits its own step events.

    Guardrails: the input check runs before any token is emitted (a
    blocked question is refused as tokens + done, never streamed); the
    output check runs in ``finish()`` on the completed answer. Streamed
    tokens already sent cannot be recalled, so each delta is also run
    through the fast regex stream scan *before* emitting — a delta that
    completes a high-precision PII shape is swallowed, not sent. See
    guardrails.stream_scan for the honest limits of that.
    """
    role = (role or "").strip().lower() or settings.GUARDRAIL_DEFAULT_ROLE
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    trace = _new_trace(question, run_id, corpus_stats, role=role)
    state: dict[str, Any] = {
        "question": question, "rewritten_question": "",
        "history": _normalize_history(history),
        "role": role, "owner_id": owner_id,
        "documents": [],
        "all_graded": [], "route": "retrieve", "grade": {}, "answer": "",
        "confidence": 0.0, "retries": 0, "trace": trace,
        "top_k": settings.TOP_K, "grounding_action": "ok",
        "web_searched": False, "force_web_search": bool(web_search_flag),
    }

    def step_running(node: str) -> None:
        emit({"type": "step", "node": node, "step": node, "status": "running"})

    def step_done(node: str) -> None:
        s = trace["steps"][-1] if trace["steps"] else {}
        emit({"type": "step", "node": node, "step": node, "status": "done",
              "detail": _step_detail(node, s), "duration_ms": s.get("duration_ms"),
              "data": dict(s)})

    def on_token(delta: str) -> None:
        emit({"type": "token", "delta": delta})

    # Stream scan state: cumulative streamed text + whether a PII shape
    # has tripped the scan (after which no further token is emitted).
    scan_state = {"text": "", "tripped": False}

    def on_token_guarded(delta: str) -> None:
        """Emit a token delta only if the cumulative streamed text still
        shows no high-precision PII shape. Checked BEFORE emitting, so
        the delta that completes a span is swallowed, not sent."""
        if scan_state["tripped"]:
            return
        scan_state["text"] += delta
        findings = stream_scan(scan_state["text"])
        if findings:
            scan_state["tripped"] = True
            trace["guardrail"]["stream_cut"] = True
            trace["guardrail"]["output_findings"] = findings
            logger.warning("Guardrail stream_cut: role=%s findings=%s",
                           role, [f["type"] for f in findings])
            return
        on_token(delta)

    def finish() -> dict:
        # Guardrail 2 — output check on the completed answer, before the
        # done event carries it to the client.
        _apply_output_guardrail(state)
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        trace["total_duration_ms"] = total_ms
        trace["route"] = state.get("route")
        trace["retries"] = int(state.get("retries") or 0)
        trace["timings_ms"]["total"] = total_ms
        docs = state.get("documents") or []
        sources = [{
            "title": d.get("title", ""), "source": d.get("source", ""),
            "score": round(float(d.get("grade", d.get("score", 0.0))), 3),
            "source_type": d.get("source_type", "local"),
            "snippet": " ".join((d.get("text") or "").split())[:800],
        } for d in docs]
        result = {
            "run_id": run_id, "answer": state.get("answer", ""),
            "confidence": float(state.get("confidence") or 0.0),
            "sources": sources, "trace": trace, "route": state.get("route"),
            "retries": int(state.get("retries") or 0),
        }
        save_run_trace(trace, owner_id=owner_id)
        return result

    def refuse_and_finish() -> None:
        """Input-blocked path: refusal as tokens + done, nothing streamed."""
        _refuse(state)
        emit({"type": "token", "delta": state["answer"]})
        emit({"type": "done", **finish()})

    # Guardrail 1 — input check, before any token is emitted.
    step_running("guardrails_input")
    verdict = _apply_input_guardrail(state, question, role)
    step_done("guardrails_input")
    if not verdict["allowed"]:
        refuse_and_finish()
        return
    step_running("contextualize_query")
    state.update(contextualize_query(state))
    step_done("contextualize_query")
    standalone = (state.get("rewritten_question") or "").strip()
    if standalone and standalone.lower() != question.strip().lower():
        step_running("guardrails_input")
        verdict = _apply_input_guardrail(state, standalone, role)
        step_done("guardrails_input")
        if not verdict["allowed"]:
            refuse_and_finish()
            return
    step_running("analyze_query")
    state.update(analyze_query(state))
    step_done("analyze_query")
    analyze_step = trace["steps"][-1] if trace["steps"] else {}
    emit({"type": "step", "node": "routing", "step": "routing",
          "status": "done",
          "detail": f"Route: {state.get('route')} ({analyze_step.get('method', 'heuristic')})",
          "data": {"route": state.get("route"),
                   "method": analyze_step.get("method", "heuristic")}})

    if state.get("route") == "clarify":
        state["answer"] = ("Could you clarify your question? Tell me what topic "
                           "or document you mean, or rephrase it as a full question.")
        state["confidence"] = 0.0
        emit({"type": "token", "delta": state["answer"]})
        emit({"type": "done", **finish()})
        return

    if state.get("route") == "retrieve":
        for _ in range(settings.MAX_RETRIES + 1):
            step_running("retrieve")
            state.update(retrieve(state))
            step_done("retrieve")
            step_running("grade_documents")
            state.update(grade_documents(state))
            step_done("grade_documents")
            if state.get("documents") or int(state.get("retries") or 0) >= settings.MAX_RETRIES:
                break
            step_running("rewrite_query")
            state.update(rewrite_query(state))
            step_done("rewrite_query")
        # Web fallback: the grader kept nothing after all local retries
        # (or the client forced web search) — search the web once, then
        # grade those chunks like local ones.
        if _web_fallback_wanted(state):
            step_running("web_search")
            state.update(web_search(state))
            step_done("web_search")
            step_running("grade_documents")
            state.update(grade_documents(state))
            step_done("grade_documents")

    def run_generate_and_ground() -> None:
        step_running("generate")
        tokens_seen = {"n": 0}

        def on_token_counted(delta: str) -> None:
            tokens_seen["n"] += 1
            on_token_guarded(delta)

        state.update(generate(state, on_token=on_token_counted))
        # Extractive fallback (or a failed LLM call) produces no token
        # events — emit the answer as one chunk so the frontend's
        # streaming path stays identical. Guarded the same way: a
        # fallback answer that trips the stream scan is not emitted;
        # the done event carries the redacted/refused answer instead.
        if tokens_seen["n"] == 0 and state.get("answer"):
            on_token_guarded(state["answer"])
        step_done("generate")
        step_running("check_grounding")
        state.update(check_grounding(state))
        step_done("check_grounding")

    run_generate_and_ground()
    if state.get("grounding_action") == "rewrite" and int(state.get("retries") or 0) < settings.MAX_RETRIES:
        step_running("rewrite_query")
        state.update(rewrite_query(state))
        step_done("rewrite_query")
        step_running("retrieve")
        state.update(retrieve(state))
        step_done("retrieve")
        step_running("grade_documents")
        state.update(grade_documents(state))
        step_done("grade_documents")
        run_generate_and_ground()
    # Grounding still failing after local retries: one web-search pass,
    # then answer from the web chunks (or the best local passages).
    if (state.get("grounding_action") == "rewrite"
            and _web_search_enabled() and not state.get("web_searched")):
        step_running("web_search")
        state.update(web_search(state))
        step_done("web_search")
        step_running("grade_documents")
        state.update(grade_documents(state))
        step_done("grade_documents")
        run_generate_and_ground()

    emit({"type": "done", **finish()})

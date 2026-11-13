"""Guardrails: input topic blocking, output PII redaction, role-based access.

Three independent layers (each can be reasoned about alone):

1. **Input guardrail** — before retrieval, the question is matched
   against blocked-topic patterns (compensation, personal data,
   credentials, plus custom phrases from config). A hit is refused with
   a generic message that reveals nothing about what was detected or
   what exists in the documents — unless the asking role is privileged
   (exec/hr by default) for that category, in which case the topic
   check passes and the role filter at retrieval decides what the role
   may actually see. Refusals are audit-logged by question hash only —
   never with the question content.
2. **Output guardrail** — before an answer is returned, it is scanned
   for PII (emails, phone numbers, PAN/Aadhaar IDs, API keys, credit
   cards). Microsoft Presidio is the detector when it is installed
   (with custom recognizers for Indian formats Presidio misses);
   otherwise a regex engine with the same patterns runs instead.
   Hits are redacted in place; an answer that *is* basically PII is
   refused outright.
3. **Role-based document access** — the enterprise control. Documents
   are ingested with an ``access_roles`` tag (default ``["employee"]``,
   i.e. public). Retrieval filters chunks to the requesting role
   *before* ranking/generation, so a restricted document's chunks can
   never reach the LLM, the sources, or the cache for that role.
   Rule: a chunk is visible when the requester's role is in its list,
   or ``"employee"`` is in its list (public). Chunks ingested before
   roles existed carry no tag and are treated as public.

Compensation figures are governed by (1) + (3), deliberately NOT by
output pattern-matching: a comp figure the role legitimately retrieved
is a legitimate answer, and a figure the role may not see can never be
retrieved in the first place (the grounding check is the backstop
against hallucinated numbers). The output scan covers identity /
contact / secret PII only.

Cache interaction (the response cache keys on question + history):
its keys MUST also include the role, or one role's answer could be
served to another. ``cache.py`` scopes keys by role for exactly this
reason; guardrail refusals (confidence 0) are never cached anyway.
"""
from __future__ import annotations

import hashlib
import logging
import re

from app.config import settings

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I can't help with that question."
PUBLIC_ROLE = "employee"

# ------------------------------------------------------------------ roles

def normalize_roles(access_roles) -> list[str]:
    """Coerce an access_roles value (list, comma string, None) to a
    sorted, de-duplicated lowercase list. None/empty means public."""
    if access_roles is None:
        return [PUBLIC_ROLE]
    if isinstance(access_roles, str):
        parts = [p.strip().lower() for p in access_roles.split(",")]
    elif isinstance(access_roles, (list, tuple, set)):
        parts = [str(p).strip().lower() for p in access_roles]
    else:
        return [PUBLIC_ROLE]
    roles = sorted({p for p in parts if p})
    return roles or [PUBLIC_ROLE]


def role_can_access(access_roles, role: str) -> bool:
    """Visible when the role is listed, or the chunk is public
    (``"employee"`` listed). Untagged (legacy) chunks are public."""
    roles = normalize_roles(access_roles)
    role = (role or "").strip().lower() or PUBLIC_ROLE
    return role in roles or PUBLIC_ROLE in roles


def chunk_access_roles(payload: dict | None) -> list[str]:
    """Access roles for a chunk payload (Qdrant payload or BM25 dict).
    Top-level ``access_roles`` wins; ``metadata.access_roles`` (the
    ingestion-time shape) is the fallback; missing means public."""
    if not isinstance(payload, dict):
        return [PUBLIC_ROLE]
    roles = payload.get("access_roles")
    if roles is None and isinstance(payload.get("metadata"), dict):
        roles = payload["metadata"].get("access_roles")
    return normalize_roles(roles)


# ------------------------------------------------------------ input guard

# Category -> (regex patterns). Word-boundary patterns are deliberately
# phrase-shaped: bare "pay" or "email" would false-positive on
# "pay attention" / "how do I email the team".
_BLOCKED_PATTERNS: dict[str, list[str]] = {
    "compensation": [
        r"\bcompensation\b", r"\bsalar(?:y|ies)\b", r"\bpayroll\b",
        r"\bremuneration\b", r"\bctc\b",
        r"\bpay\s+(?:bracket|scale|package|grade|band|rise|hike)\b",
        # Bare "pay" is only a signal in comp phrasings ("CEO's pay",
        # "what is the pay", "how much is the pay") — never on its own,
        # or "pay attention" would false-positive.
        r"'s\s+pay\b",
        r"\bwhat(?:'s| is)\b.{0,30}\bpay\b",
        r"\bhow\s+much\b.{0,40}\bpay\b",
        r"\bpay\s+(?:is|are|was|were|does)\b",
        r"\bwhat\s+(?:does|do|did)\b.{0,30}\b(?:earn|make|paid)\b",
        r"\bhow\s+much\b.{0,40}\b(?:earn|make|paid|salary)\b",
        r"\btake[\s-]?home\b", r"\bstock\s+options?\b",
        r"\bincome\b", r"\bwages\b", r"\bincrement\b", r"\bappraisal\b",
        r"\bhow\s+much\s+(?:does|do|is|are|did)\b.{0,40}\b(?:earn|make|paid|get\s+paid)\b",
        r"\b(?:earn|earns|earning|makes|paid)\b.{0,20}\b(?:per\s+annum|per\s+year|a\s+year|monthly|per\s+month)\b",
        r"\bbonus\b",
    ],
    "personal_data": [
        r"\bphone\s*(?:number|no\.?)?\b", r"\bmobile\s*(?:number|no\.?)\b",
        r"\bcontact\s*(?:number|details|info)\b",
        r"\bemail\s*(?:address|id|of)\b", r"\bmail\s*id\b",
        r"\bhome\s+address\b", r"\bresidential\s+address\b",
        r"\bdate\s+of\s+birth\b", r"\bdob\b",
        r"\bpersonal\s+(?:details|data|information)\b",
        r"\baadhaar\b", r"\bpan\s*(?:card|number)\b",
        r"\bpassport\s*(?:number|details)\b",
        r"\bbank\s*(?:account|details)\b",
        r"\baddress\s+of\b", r"\bphone\s+of\b",
        r"'s\s+(?:email|phone|number|address)\b",
    ],
    "credentials": [
        r"\bpassword\b", r"\bapi\s*key\b", r"\bapikey\b",
        r"\bsecret\s*(?:key|token)?\b", r"\baccess\s*key\b",
        r"\bprivate\s*key\b", r"\bauth\s*token\b", r"\baccess\s*token\b",
        r"\bcredentials\b", r"\bclient\s*secret\b",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _BLOCKED_PATTERNS.items()
}


def _enabled_categories() -> list[str]:
    raw = getattr(settings, "GUARDRAIL_BLOCKED_TOPICS", "") or ""
    cats = [c.strip().lower() for c in raw.split(",") if c.strip()]
    return [c for c in cats if c in _BLOCKED_PATTERNS]


def _custom_phrases() -> list[str]:
    raw = getattr(settings, "GUARDRAIL_BLOCKED_PHRASES", "") or ""
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _privileged_roles() -> set[str]:
    """Roles that may ask about otherwise-blocked topics (compensation,
    personal data) — their answers are still governed by the role
    filter, so they can only retrieve documents their role may see.
    Credentials are blocked for every role, privileged or not."""
    raw = getattr(settings, "GUARDRAIL_PRIVILEGED_ROLES", "") or ""
    return {r.strip().lower() for r in raw.split(",") if r.strip()}


# Categories a privileged role may ask about (still role-filtered at
# retrieval). Everything else blocks for every role.
_PRIVILEGED_CATEGORIES = {"compensation", "personal_data"}


def check_input(question: str, role: str | None = None) -> dict:
    """Input guardrail. Returns {allowed, category, matched, reason,
    privileged_bypass}.

    A blocked topic is refused for ordinary roles, but a privileged
    role (exec/hr by default) passes the topic check — the role filter
    at retrieval is what then decides which documents it may actually
    see. That is the whole point of roles: an exec *may* ask about CEO
    pay and get the figure from the exec-tagged doc, while an employee
    asking the same question is refused at the door and never retrieves
    the restricted chunks. Credentials are blocked for every role.
    ``privileged_bypass`` names the category the role bypassed, if any
    (audit trail — the question still matched a blocked pattern).
    """
    if not getattr(settings, "GUARDRAILS_ENABLED", True):
        return {"allowed": True, "category": None, "matched": None,
                "reason": None, "privileged_bypass": None}
    text = question or ""
    lowered = text.lower()
    privileged = (role or "").strip().lower() in _privileged_roles()
    bypass: str | None = None
    for category in _enabled_categories():
        matched_text = None
        for pattern in _COMPILED[category]:
            m = pattern.search(text)
            if m:
                matched_text = m.group(0)
                break
        if matched_text is None:
            continue
        if category in _PRIVILEGED_CATEGORIES and privileged:
            # Matched a blocked topic, but this role is allowed to ask
            # about it — retrieval (role-filtered) decides what it sees.
            if bypass is None:
                bypass = category
            continue
        log_guardrail_event("input_block", role, category, question)
        return {"allowed": False, "category": category,
                "matched": matched_text,
                "reason": f"Blocked topic: {category}",
                "privileged_bypass": bypass}
    for phrase in _custom_phrases():
        if phrase and phrase in lowered:
            log_guardrail_event("input_block", role, "custom", question)
            return {"allowed": False, "category": "custom",
                    "matched": phrase, "reason": "Blocked topic: custom",
                    "privileged_bypass": bypass}
    return {"allowed": True, "category": None, "matched": None,
            "reason": None, "privileged_bypass": bypass}


def log_guardrail_event(kind: str, role: str | None, category: str | None,
                        question: str | None = None) -> None:
    """Audit log for a guardrail decision. Only a hash of the question
    is recorded — never the question or answer content."""
    qhash = ""
    if question:
        qhash = hashlib.sha256(
            re.sub(r"\s+", " ", question.strip().lower()).encode("utf-8")
        ).hexdigest()[:12]
    logger.warning("Guardrail %s: role=%s category=%s question_sha=%s",
                   kind, role or "?", category or "-", qhash or "-")


# ----------------------------------------------------------- output guard

# Config type name -> Presidio entity type.
_PII_MAP = {
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "pan": "PAN",
    "aadhaar": "AADHAAR",
    "api_key": "API_KEY",
    "credit_card": "CREDIT_CARD",
    "person": "PERSON",
}
_PII_REVERSE = {v: k for k, v in _PII_MAP.items()}

# Regex engine (fallback when Presidio is unavailable, and the fast
# streaming scanner). Same entity shapes as the Presidio recognizers.
_REGEX_PATTERNS: dict[str, list[str]] = {
    "email": [r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"],
    "phone": [
        # Indian mobile, with optional +91 / spaces / dashes.
        r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b",
        # Generic international, 10+ digits — only as a backstop.
        r"\+\d[\d\s\-()]{8,}\d",
    ],
    "pan": [r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"],
    "aadhaar": [r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"],
    "api_key": [
        r"\bsk-[A-Za-z0-9]{20,}\b",
        r"\bghp_[A-Za-z0-9]{20,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"(?i)(?:api[_-]?key|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    ],
    "credit_card": [r"\b(?:\d[ \-]?){13,19}\b"],
}

# Fast streaming scan: only the highest-precision shapes, so a live
# token stream is never cut on a false positive.
_STREAM_PATTERNS: dict[str, list[str]] = {
    "email": _REGEX_PATTERNS["email"],
    "pan": _REGEX_PATTERNS["pan"],
    "aadhaar": _REGEX_PATTERNS["aadhaar"],
    "phone": [_REGEX_PATTERNS["phone"][0]],
    "api_key": _REGEX_PATTERNS["api_key"],
}

_COMPILED_REGEX: dict[str, list[re.Pattern]] = {
    k: [re.compile(p) for p in pats] for k, pats in _REGEX_PATTERNS.items()
}
_COMPILED_STREAM: dict[str, list[re.Pattern]] = {
    k: [re.compile(p) for p in pats] for k, pats in _STREAM_PATTERNS.items()
}


def _enabled_pii_types() -> list[str]:
    raw = getattr(settings, "GUARDRAIL_PII_TYPES", "") or ""
    types = [t.strip().lower() for t in raw.split(",") if t.strip()]
    return [t for t in types if t in _PII_MAP]


def _luhn_ok(digits: str) -> bool:
    """Credit-card-shaped digit runs must pass Luhn, or they are just
    long numbers (order ids, timestamps) — never redact those."""
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _regex_scan(text: str, types: list[str]) -> list[dict]:
    findings: list[dict] = []
    for pii_type in types:
        for pattern in _COMPILED_REGEX.get(pii_type, []):
            for m in pattern.finditer(text or ""):
                if pii_type == "credit_card" and not _luhn_ok(m.group(0)):
                    continue
                findings.append({"type": pii_type, "start": m.start(),
                                 "end": m.end(), "score": 0.95})
    return _dedupe_findings(findings)


def _dedupe_findings(findings: list[dict]) -> list[dict]:
    """Drop overlapping spans, keeping the higher score (then longer)."""
    ordered = sorted(findings,
                     key=lambda f: (f["start"], -f["score"], -(f["end"] - f["start"])))
    kept: list[dict] = []
    for f in ordered:
        if any(f["start"] < k["end"] and k["start"] < f["end"] for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f["start"])


_analyzer = None
_analyzer_failed = False


def _build_analyzer():
    """Presidio analyzer with custom recognizers for Indian formats.
    Built once, lazily — importing guardrails must never fail when
    Presidio (or its spaCy model) is missing; the regex engine covers it.
    """
    global _analyzer, _analyzer_failed
    if _analyzer is not None:
        return _analyzer
    if _analyzer_failed:
        return None
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        nlp = SpacyNlpEngine(
            models=[{"lang_code": "en", "model_name": "en_core_web_sm"}])
        analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PAN", patterns=[
                Pattern(name="pan", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                        score=0.9)]))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="AADHAAR", patterns=[
                Pattern(name="aadhaar", regex=r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
                        score=0.85)]))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PHONE_NUMBER", patterns=[
                Pattern(name="in_mobile",
                        regex=r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b",
                        score=0.9)]))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="API_KEY", patterns=[
                Pattern(name="sk_key", regex=r"\bsk-[A-Za-z0-9]{20,}\b",
                        score=0.95),
                Pattern(name="gh_key", regex=r"\bghp_[A-Za-z0-9]{20,}\b",
                        score=0.95),
                Pattern(name="aws_key", regex=r"\bAKIA[0-9A-Z]{16}\b",
                        score=0.95),
                Pattern(name="kv_key",
                        regex=r"(?i)(?:api[_-]?key|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
                        score=0.9)]))
        _analyzer = analyzer
        logger.info("Guardrails: Presidio PII analyzer ready "
                    "(en_core_web_sm + PAN/Aadhaar/IN-phone/API-key).")
        return analyzer
    except Exception as exc:  # noqa: BLE001 - any failure -> regex engine
        _analyzer_failed = True
        logger.warning("Guardrails: Presidio unavailable (%s); using regex "
                       "PII scan.", exc)
        return None


def pii_engine() -> str:
    """Which output engine would run: 'presidio' if its packages and the
    spaCy model are importable, else 'regex'. No engine is built here."""
    try:
        import importlib.util

        if importlib.util.find_spec("presidio_analyzer") is None:
            return "regex"
        if importlib.util.find_spec("en_core_web_sm") is None:
            return "regex"
        return "presidio"
    except Exception:  # noqa: BLE001 - import probing must not raise
        return "regex"


def _presidio_scan(text: str, types: list[str]) -> list[dict] | None:
    analyzer = _build_analyzer()
    if analyzer is None:
        return None
    wanted = {_PII_MAP[t] for t in types}
    try:
        results = analyzer.analyze(text=text or "", language="en")
    except Exception as exc:  # noqa: BLE001 - fall back to regex
        logger.warning("Guardrails: Presidio analyze failed (%s); using "
                       "regex scan.", exc)
        return None
    findings = []
    for r in results:
        pii_type = _PII_REVERSE.get(r.entity_type)
        if pii_type is None or pii_type not in types:
            continue
        if float(r.score) < 0.5:
            continue
        if pii_type == "credit_card" and not _luhn_ok(
                (text or "")[r.start:r.end]):
            continue
        findings.append({"type": pii_type, "start": int(r.start),
                         "end": int(r.end), "score": round(float(r.score), 3)})
    return _dedupe_findings(findings)


def _redact(text: str, findings: list[dict]) -> str:
    out = text or ""
    for f in sorted(findings, key=lambda f: f["start"], reverse=True):
        out = out[:f["start"]] + f"[REDACTED:{f['type'].upper()}]" + out[f["end"]:]
    return out


def check_output(answer: str, role: str | None = None) -> dict:
    """Output guardrail. Returns {status, answer, findings}.

    status: "pass" (no PII), "redact" (spans replaced), "block" (the
    answer is basically PII — refuse it outright). ``role`` is recorded
    for the audit log; redaction itself is role-independent (contact /
    ID / secret data is never served raw, even to privileged roles).
    """
    if not getattr(settings, "GUARDRAILS_ENABLED", True):
        return {"status": "pass", "answer": answer, "findings": []}
    types = _enabled_pii_types()
    if not types or not (answer or "").strip():
        return {"status": "pass", "answer": answer, "findings": []}
    findings = _presidio_scan(answer, types)
    if findings is None:
        findings = _regex_scan(answer, types)
    if not findings:
        return {"status": "pass", "answer": answer, "findings": []}
    covered = sum(f["end"] - f["start"] for f in findings)
    remaining = len(answer or "") - covered
    log_guardrail_event("output_" + ("block" if remaining < 20 or
                                     covered / max(len(answer or ""), 1) > 0.6
                                     else "redact"),
                        role, ",".join(sorted({f["type"] for f in findings})))
    if remaining < 20 or covered / max(len(answer or ""), 1) > 0.6:
        return {"status": "block", "answer": REFUSAL_MESSAGE,
                "findings": findings}
    return {"status": "redact", "answer": _redact(answer, findings),
            "findings": findings}


def stream_scan(cumulative_text: str) -> list[dict]:
    """Fast regex scan for the live token stream (Presidio is too slow
    per delta). Returns findings if a high-precision PII shape has
    completed in the text so far; the caller stops emitting tokens.
    A delta can still carry a full span before the scan sees it — the
    input guardrail + role filter are the real controls for streaming;
    this only limits the bleed. Returns [] when guardrails are off.
    """
    if not getattr(settings, "GUARDRAILS_ENABLED", True):
        return []
    if not getattr(settings, "GUARDRAIL_STREAM_SCAN", True):
        return []
    types = [t for t in _enabled_pii_types() if t in _STREAM_PATTERNS]
    return _regex_scan(cumulative_text or "", types)


def guardrail_status() -> dict:
    """Health-endpoint summary (no secrets, no content)."""
    return {
        "enabled": bool(getattr(settings, "GUARDRAILS_ENABLED", True)),
        "default_role": getattr(settings, "GUARDRAIL_DEFAULT_ROLE",
                                PUBLIC_ROLE),
        "blocked_topics": _enabled_categories(),
        "pii_types": _enabled_pii_types(),
        "pii_engine": pii_engine(),
    }

"""
ClinicalEdge Security & Guardrails Module
==========================================
Implements three layers of defense:
  1. PHI / PII detection  — blocks queries containing patient-identifiable info
  2. Domain scope guard   — rejects questions unrelated to pharma / biotech
  3. Rate limiter         — caps outbound API calls at 3 requests/sec per source
  4. Output sanitizer     — strips any PHI that leaked into agent responses

Design principle: fail-closed.  Any guardrail exception blocks the request
and returns a safe explanatory message rather than allowing the query through.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("clinicaledge.security")

# ---------------------------------------------------------------------------
# 1. PHI / PII Detection
# ---------------------------------------------------------------------------

# Patterns that may indicate Protected Health Information
_PHI_PATTERNS: dict[str, re.Pattern] = {
    "ssn":        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "dob":        re.compile(
        r"\b(born|dob|date\s+of\s+birth)\s*[:\-]?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        re.IGNORECASE,
    ),
    "mrn":        re.compile(r"\b(mrn|medical\s+record\s+(number|no\.?))\s*[:#]?\s*\d{4,}\b", re.IGNORECASE),
    "npi":        re.compile(r"\b(npi)\s*[:#]?\s*\d{10}\b", re.IGNORECASE),
    "phone":      re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "email":      re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "patient_name": re.compile(
        r"\b(patient|pt\.?)\s+(name|id)\s*[:\-]?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        re.IGNORECASE,
    ),
    "full_name_prefix": re.compile(
        r"\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"
    ),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "zip_plus4":   re.compile(r"\b\d{5}-\d{4}\b"),
}

# Additional high-risk keywords that suggest clinical patient data
_PHI_KEYWORDS = frozenset({
    "patient", "subject id", "participant id", "case number",
    "hospital record", "health record", "diagnosis code", "icd-10",
    "insurance id", "beneficiary", "hipaa",
})


@dataclass
class GuardrailResult:
    """Returned by every guardrail check. Carry the verdict and a safe message."""
    allowed: bool
    reason: str = ""
    matched_rule: str = ""


def check_phi(text: str) -> GuardrailResult:
    """
    Scan *text* for Protected Health Information patterns.

    Returns GuardrailResult(allowed=False) if any PHI indicator is found.
    All checks are performed on a lower-cased copy to catch mixed-case leaks,
    but the original text is used for regex matching where case matters.
    """
    text_lower = text.lower()

    # Keyword pre-scan (fast O(n) string search before heavier regex)
    for kw in _PHI_KEYWORDS:
        if kw in text_lower:
            logger.warning("PHI keyword detected: '%s'", kw)
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"Query blocked: potential PHI keyword detected ('{kw}'). "
                    "ClinicalEdge processes population-level pharma intelligence only — "
                    "not individual patient data."
                ),
                matched_rule=f"phi_keyword:{kw}",
            )

    # Regex pattern scan
    for name, pattern in _PHI_PATTERNS.items():
        if pattern.search(text):
            logger.warning("PHI pattern detected: %s", name)
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"Query blocked: potential PHI detected (pattern: {name}). "
                    "Please submit population-level or drug-level queries only."
                ),
                matched_rule=f"phi_pattern:{name}",
            )

    return GuardrailResult(allowed=True)


# ---------------------------------------------------------------------------
# 2. Domain Scope Validator
# ---------------------------------------------------------------------------

# Terms that firmly anchor a query to the pharma / biotech domain
_PHARMA_ANCHOR_TERMS = frozenset({
    # Drug & molecule types
    "drug", "compound", "molecule", "mab", "inhibitor", "antibody", "biologic",
    "small molecule", "peptide", "gene therapy", "cell therapy", "car-t",
    "vaccine", "agonist", "antagonist", "modulator",
    # Disease / indication
    "cancer", "tumor", "oncology", "nsclc", "crc", "aml", "myeloma",
    "cardiovascular", "diabetes", "autoimmune", "rare disease", "indication",
    "therapeutic area",
    # Development stages
    "clinical trial", "phase 1", "phase 2", "phase 3", "phase i", "phase ii",
    "phase iii", "fda", "ema", "nda", "bla", "ind", "pma", "approval",
    "regulatory", "biomarker", "endpoint", "efficacy", "safety",
    # Scientific / research
    "mechanism of action", "moa", "pharmacokinetics", "pk", "pd",
    "pharmacodynamics", "target", "receptor", "pathway", "mutation",
    "kras", "egfr", "her2", "pd-l1", "brca",
    # Industry
    "pharma", "biotech", "pharmaceutical", "sponsor", "pipeline", "portfolio",
    "competitive landscape", "market",
    # Databases
    "pubmed", "clinicaltrials", "fda", "openfda", "ncbi",
})

# Minimum number of anchor terms a query must contain
_MIN_ANCHOR_HITS = 1


def check_domain_scope(text: str) -> GuardrailResult:
    """
    Verify the query is pharma / biotech related.

    We require at least one pharma anchor term.  This is deliberately lenient
    (threshold = 1) because real queries are often short.  Increase
    _MIN_ANCHOR_HITS for stricter enforcement.
    """
    text_lower = text.lower()
    hits = [term for term in _PHARMA_ANCHOR_TERMS if term in text_lower]

    if len(hits) >= _MIN_ANCHOR_HITS:
        logger.debug("Domain scope OK — matched terms: %s", hits[:5])
        return GuardrailResult(allowed=True)

    logger.warning("Out-of-scope query rejected (no pharma anchors found).")
    return GuardrailResult(
        allowed=False,
        reason=(
            "Query out of scope.  ClinicalEdge answers pharma / biotech competitive "
            "intelligence questions only (e.g., drug pipelines, clinical trials, "
            "FDA approvals).  Please rephrase your question."
        ),
        matched_rule="domain_scope:no_anchor",
    )


# ---------------------------------------------------------------------------
# 3. Rate Limiter — token-bucket per API source
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Per-source sliding-window rate limiter.

    Allows at most *max_calls* requests within a *window_seconds* rolling window.
    All state is in-memory (per-process).  For distributed deployments, replace
    the deque with a Redis sorted set.

    Usage (async):
        limiter = RateLimiter(max_calls=3, window_seconds=1.0)
        await limiter.acquire("pubmed")
        response = await call_pubmed_api(...)
    """

    def __init__(self, max_calls: int = 3, window_seconds: float = 1.0) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        # source_name → deque of call timestamps (monotonic)
        self._windows: dict[str, deque] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, source: str) -> None:
        """
        Block until a request slot is available for *source*.
        Raises RuntimeError if waiting would exceed 30 s (circuit-breaker).
        """
        lock = self._locks[source]
        async with lock:
            window_deq = self._windows[source]
            deadline = time.monotonic() + 30.0  # circuit-breaker

            while True:
                now = time.monotonic()
                if now > deadline:
                    raise RuntimeError(
                        f"Rate-limiter circuit-breaker tripped for source '{source}'. "
                        "Too many concurrent requests."
                    )

                # Evict timestamps older than the rolling window
                while window_deq and now - window_deq[0] >= self.window:
                    window_deq.popleft()

                if len(window_deq) < self.max_calls:
                    window_deq.append(now)
                    logger.debug("Rate-limiter acquired slot for '%s' (%d/%d)",
                                 source, len(window_deq), self.max_calls)
                    return

                # Calculate sleep time until the oldest slot expires
                sleep_for = self.window - (now - window_deq[0]) + 0.01
                logger.debug("Rate-limiter throttling '%s' — sleeping %.3fs", source, sleep_for)
                await asyncio.sleep(sleep_for)


# Singleton instance shared across all MCP servers
_global_rate_limiter = RateLimiter(max_calls=3, window_seconds=1.0)


def get_rate_limiter() -> RateLimiter:
    """Return the shared rate-limiter singleton."""
    return _global_rate_limiter


# ---------------------------------------------------------------------------
# 4. Output Sanitizer
# ---------------------------------------------------------------------------

# Patterns to redact from agent output (defense-in-depth)
_OUTPUT_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED-EMAIL]"),
    (re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED-PHONE]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED-CC]"),
]


def sanitize_output(text: str) -> str:
    """
    Strip any PHI patterns that may have leaked into the agent's response.

    This is a secondary defense layer — the primary layer is check_phi() on
    the input.  Sanitization ensures that even if an API returns unexpected
    patient-level data, it is scrubbed before reaching the analyst.
    """
    for pattern, replacement in _OUTPUT_REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# 5. Composite gate — run all checks in order
# ---------------------------------------------------------------------------

def run_all_guardrails(query: str) -> GuardrailResult:
    """
    Run PHI check → domain scope check in sequence.

    Returns the first failing GuardrailResult, or GuardrailResult(allowed=True)
    if all checks pass.  Call this once at the orchestrator entry point before
    delegating to any specialist agent.
    """
    for check_fn in (check_phi, check_domain_scope):
        result = check_fn(query)
        if not result.allowed:
            return result
    return GuardrailResult(allowed=True, reason="All guardrails passed.")

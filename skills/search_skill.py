"""
Shared Search Skill
====================
Utility functions used by all three specialist agents for query construction,
result normalization, and relevance scoring.

Kept as a standalone module (not an ADK tool) so specialist agents can import
it directly without adding round-trip latency.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Query Construction Helpers
# ---------------------------------------------------------------------------

def build_pubmed_query(
    drug_class: str,
    indication: str,
    synonyms: list[str] | None = None,
    min_year: int | None = None,
) -> str:
    """
    Build an optimized PubMed Boolean search string.

    Combines drug class, indication, and optional synonyms using AND/OR logic
    following PubMed best practices (MeSH terms + free-text fallback).

    Example:
        build_pubmed_query("KRAS G12C inhibitor", "NSCLC", synonyms=["sotorasib", "adagrasib"])
        → '("KRAS G12C"[Title/Abstract] OR "KRASG12C"[Title/Abstract]) AND ("NSCLC"[Title/Abstract] OR "non-small cell lung cancer"[MeSH Terms]) AND (sotorasib[Title/Abstract] OR adagrasib[Title/Abstract])'
    """
    parts = []

    # Drug class term (try MeSH + free text)
    drug_terms = [f'"{drug_class}"[Title/Abstract]']
    if synonyms:
        drug_terms += [f'"{s}"[Title/Abstract]' for s in synonyms]
    parts.append(f"({' OR '.join(drug_terms)})")

    # Indication term
    indication_terms = [
        f'"{indication}"[Title/Abstract]',
        f'"{indication}"[MeSH Terms]',
    ]
    parts.append(f"({' OR '.join(indication_terms)})")

    # Limit to clinical/drug literature (exclude reviews of unrelated topics)
    parts.append('("clinical trial"[pt] OR "drug therapy"[MeSH] OR "comparative study"[pt] OR "review"[pt])')

    query = " AND ".join(parts)

    if min_year:
        query += f" AND {min_year}:3000[pdat]"

    return query


def build_trial_query(drug_class: str, indication: str, synonyms: list[str] | None = None) -> str:
    """
    Build a ClinicalTrials.gov search query string.

    ClinicalTrials.gov v2 uses a simpler keyword query — we combine
    the most specific terms for relevance.
    """
    terms = [drug_class, indication]
    if synonyms:
        terms.extend(synonyms[:2])  # keep it focused
    return " ".join(terms)


# ---------------------------------------------------------------------------
# Relevance Scoring
# ---------------------------------------------------------------------------

# Keywords that boost relevance for competitive intelligence
_CI_KEYWORDS = [
    "competitive", "landscape", "comparison", "versus", "approval", "fda",
    "efficacy", "survival", "overall survival", "progression-free", "response rate",
    "phase 3", "phase iii", "randomized", "pivotal", "head-to-head",
]


def score_article_relevance(article: dict[str, Any], drug_class: str, indication: str) -> float:
    """
    Score a PubMed article [0.0–1.0] for competitive intelligence relevance.

    Factors:
      - Title/abstract contains drug class name           (+0.3)
      - Title/abstract contains indication name            (+0.3)
      - Contains competitive intelligence keywords         (+0.2)
      - Publication recency (exponential decay over 5yr)  (+0.2)
    """
    score = 0.0
    text = f"{article.get('title', '')} {article.get('abstract_snippet', '')}".lower()

    if drug_class.lower() in text:
        score += 0.30
    if indication.lower() in text:
        score += 0.30

    ci_hits = sum(1 for kw in _CI_KEYWORDS if kw in text)
    score += min(ci_hits * 0.05, 0.20)

    # Recency: parse year from pub_date, decay over 5 years
    pub_date = article.get("pub_date", "")
    year_match = re.search(r"\b(20\d{2})\b", pub_date)
    if year_match:
        pub_year = int(year_match.group(1))
        current_year = datetime.now().year
        age = max(0, current_year - pub_year)
        recency_score = max(0.0, 0.20 * (1 - age / 5))
        score += recency_score

    return round(min(score, 1.0), 3)


def score_trial_relevance(trial: dict[str, Any], drug_class: str, indication: str) -> float:
    """
    Score a clinical trial [0.0–1.0] for competitive intelligence relevance.

    Factors:
      - Condition matches indication                       (+0.35)
      - Intervention matches drug class                   (+0.35)
      - Trial is in Phase 2/3 (more strategically useful) (+0.20)
      - Trial is active/recruiting                        (+0.10)
    """
    score = 0.0

    conditions_text = " ".join(trial.get("conditions", [])).lower()
    interventions_text = " ".join(trial.get("interventions", [])).lower()
    brief_summary = trial.get("brief_summary", "").lower()

    if indication.lower() in conditions_text or indication.lower() in brief_summary:
        score += 0.35
    if drug_class.lower() in interventions_text or drug_class.lower() in brief_summary:
        score += 0.35

    phase = trial.get("phase", "").lower()
    if "phase 3" in phase or "phase iii" in phase:
        score += 0.20
    elif "phase 2" in phase or "phase ii" in phase:
        score += 0.10

    status = trial.get("status", "").lower()
    if "recruiting" in status or "active" in status:
        score += 0.10

    return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Result Normalization
# ---------------------------------------------------------------------------

def normalize_articles(articles: list[dict], drug_class: str, indication: str) -> list[dict]:
    """
    Score, sort, and clean PubMed article results for downstream synthesis.
    Returns articles sorted by relevance (highest first).
    """
    for a in articles:
        a["relevance_score"] = score_article_relevance(a, drug_class, indication)

    return sorted(articles, key=lambda x: x["relevance_score"], reverse=True)


def normalize_trials(trials: list[dict], drug_class: str, indication: str) -> list[dict]:
    """
    Score, sort, and clean ClinicalTrials results for downstream synthesis.
    Returns trials sorted by relevance (highest first).
    """
    for t in trials:
        t["relevance_score"] = score_trial_relevance(t, drug_class, indication)

    return sorted(trials, key=lambda x: x["relevance_score"], reverse=True)


# ---------------------------------------------------------------------------
# Entity Extraction (light-weight, no heavy NLP dependency)
# ---------------------------------------------------------------------------

# Common drug class synonyms for known targets
_KNOWN_SYNONYMS: dict[str, list[str]] = {
    "kras g12c": ["sotorasib", "adagrasib", "divarasib", "glecirasib", "mrtx849"],
    "egfr": ["erlotinib", "gefitinib", "osimertinib", "afatinib"],
    "pd-l1": ["atezolizumab", "durvalumab", "avelumab"],
    "pd-1": ["pembrolizumab", "nivolumab", "cemiplimab"],
    "her2": ["trastuzumab", "pertuzumab", "lapatinib", "tucatinib"],
    "alk": ["crizotinib", "alectinib", "brigatinib", "lorlatinib"],
    "bcr-abl": ["imatinib", "dasatinib", "nilotinib", "ponatinib"],
}


def extract_drug_synonyms(drug_class: str) -> list[str]:
    """
    Return known drug name synonyms for a drug class/target.

    Falls back to an empty list for unknown targets — specialist agents
    can still run without synonyms.
    """
    drug_lower = drug_class.lower()
    for key, synonyms in _KNOWN_SYNONYMS.items():
        if key in drug_lower:
            return synonyms
    return []


def clean_text(text: str, max_length: int = 300) -> str:
    """
    Strip extra whitespace and truncate text for LLM token efficiency.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > max_length:
        return cleaned[:max_length] + "…"
    return cleaned

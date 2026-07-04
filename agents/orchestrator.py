"""
ClinicalEdge Root Orchestrator Agent
======================================
The central coordinator of the ClinicalEdge multi-agent system.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │           ORCHESTRATOR (this module)            │
  │  1. Validate & guardrail the incoming query     │
  │  2. Extract: drug_class, indication, time_window │
  │  3. Fan out to 3 specialist agents (parallel)   │
  │  4. Aggregate all structured results            │
  │  5. Call synthesis skill → IntelligenceReport   │
  │  6. Format & return the 5-section report        │
  └─────────────────────────────────────────────────┘
         │              │              │
    Literature       Trial         Regulatory
      Scout          Monitor         Watch
    (PubMed)   (ClinicalTrials)   (openFDA)

ADK integration:
  - The orchestrator is itself an LlmAgent that receives the analyst's
    natural-language query and coordinates the sub-agents.
  - For the Kaggle notebook demo, `run_orchestrator_direct()` bypasses
    the ADK runner and invokes sub-agents directly — no subprocess needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("clinicaledge.orchestrator")

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """You are the ClinicalEdge Orchestrator, the master coordinator of a pharmaceutical competitive intelligence system.

Your workflow for every analyst query:

STEP 1 — ENTITY EXTRACTION
Parse the natural-language query and extract:
  - drug_class: the drug target or class (e.g., "KRAS G12C inhibitor")
  - indication: the disease/condition (e.g., "non-small cell lung cancer", "NSCLC")
  - time_window: any specified time frame (e.g., "as of 2024", "last 3 years")
  - synonyms: list known drug names in this class

STEP 2 — PARALLEL DELEGATION
Simultaneously dispatch to all three specialist agents:
  - Literature Scout: query PubMed for recent research papers
  - Trial Monitor: query ClinicalTrials.gov for active/completed trials
  - Regulatory Watch: query openFDA for approval records and safety data

STEP 3 — AGGREGATION
Collect all three structured JSON responses and merge them.

STEP 4 — SYNTHESIS
Generate a professional 5-section intelligence report:
  1. Research Landscape
  2. Clinical Pipeline
  3. Regulatory Status
  4. Competitive Summary
  5. Strategic Outlook

Always cite specific evidence: PMIDs, NCT IDs, NDA/BLA numbers.
Be analytical, not just descriptive. Identify patterns and competitive implications."""

# ---------------------------------------------------------------------------
# Entity extraction (regex + heuristics — no external NLP dependency)
# ---------------------------------------------------------------------------

# Known drug class → indication patterns for better entity extraction
_KNOWN_CLASSES: dict[str, dict[str, Any]] = {
    r"kras\s*g12c": {
        "drug_class": "KRAS G12C inhibitor",
        "synonyms": ["sotorasib", "adagrasib", "divarasib", "glecirasib"],
    },
    r"egfr\s+inhibitor|egfr\s+tki": {
        "drug_class": "EGFR inhibitor",
        "synonyms": ["erlotinib", "gefitinib", "osimertinib", "afatinib"],
    },
    r"pd-?l1\s+inhibitor|pd-?l1\s+checkpoint": {
        "drug_class": "PD-L1 inhibitor",
        "synonyms": ["atezolizumab", "durvalumab", "avelumab"],
    },
    r"pd-?1\s+inhibitor|checkpoint\s+inhibitor": {
        "drug_class": "PD-1 inhibitor",
        "synonyms": ["pembrolizumab", "nivolumab", "cemiplimab"],
    },
    r"alk\s+inhibitor|alk\s+tki": {
        "drug_class": "ALK inhibitor",
        "synonyms": ["crizotinib", "alectinib", "brigatinib", "lorlatinib"],
    },
    r"her2\s+inhibitor|her2\s+targeted": {
        "drug_class": "HER2-targeted therapy",
        "synonyms": ["trastuzumab", "pertuzumab", "lapatinib", "tucatinib"],
    },
    r"car[\s-]t|car\s+t[\s-]cell": {
        "drug_class": "CAR-T cell therapy",
        "synonyms": ["tisagenlecleucel", "axicabtagene", "lisocabtagene"],
    },
}

# Indication normalization map
_INDICATION_MAP: dict[str, str] = {
    r"nsclc|non[\s-]small[\s-]cell\s+lung": "non-small cell lung cancer (NSCLC)",
    r"sclc|small[\s-]cell\s+lung":           "small cell lung cancer (SCLC)",
    r"crc|colorectal|colon\s+cancer":         "colorectal cancer (CRC)",
    r"aml|acute\s+myeloid":                  "acute myeloid leukemia (AML)",
    r"cll|chronic\s+lymphocytic":             "chronic lymphocytic leukemia (CLL)",
    r"multiple\s+myeloma":                    "multiple myeloma",
    r"pancreatic\s+cancer|pdac":              "pancreatic ductal adenocarcinoma (PDAC)",
    r"breast\s+cancer":                       "breast cancer",
    r"prostate\s+cancer":                     "prostate cancer",
    r"melanoma":                              "melanoma",
}


@dataclass
class QueryEntities:
    """Structured entities extracted from the analyst's natural-language query."""
    original_query: str
    drug_class:     str
    indication:     str
    time_window:    str
    synonyms:       list[str]
    min_year:       int


def extract_entities(query: str) -> QueryEntities:
    """
    Extract structured entities from a natural-language competitive intelligence query.

    Uses pattern matching against known drug class and indication vocabularies.
    Falls back to using the raw query text for unknown targets.

    Args:
        query: Analyst's natural-language question.

    Returns:
        QueryEntities with extracted drug_class, indication, time_window, etc.
    """
    query_lower = query.lower()

    # --- Drug class detection ---
    drug_class = ""
    synonyms: list[str] = []

    for pattern, info in _KNOWN_CLASSES.items():
        if re.search(pattern, query_lower):
            drug_class = info["drug_class"]
            synonyms = info["synonyms"]
            break

    if not drug_class:
        # Fallback: try to extract anything between "for" and "in"
        match = re.search(r"\bfor\s+([^.?]+?)\s+in\b", query_lower)
        drug_class = match.group(1).strip().title() if match else query[:50]

    # --- Indication detection ---
    indication = ""
    for pattern, normalized in _INDICATION_MAP.items():
        if re.search(pattern, query_lower):
            indication = normalized
            break

    if not indication:
        # Fallback: extract text after "in " near the end of the query
        match = re.search(r"\bin\s+([^.?]+?)(?:\s+as\s+of|\s+from|\s+since|$|\?)", query_lower)
        indication = match.group(1).strip().title() if match else "oncology"

    # --- Time window detection ---
    time_window = "recent (last 5 years)"
    year_match = re.search(r"\b(as\s+of\s+)?(20\d{2})\b", query_lower)
    if year_match:
        year = int(year_match.group(2))
        time_window = f"as of {year}"
        min_year = year - 4  # 5-year lookback from the specified year
    else:
        min_year = 2020

    logger.info(
        "Entity extraction: drug_class='%s', indication='%s', time_window='%s', synonyms=%s",
        drug_class, indication, time_window, synonyms,
    )

    return QueryEntities(
        original_query=query,
        drug_class=drug_class,
        indication=indication,
        time_window=time_window,
        synonyms=synonyms,
        min_year=min_year,
    )


# ---------------------------------------------------------------------------
# ADK Orchestrator agent definition
# ---------------------------------------------------------------------------

def create_orchestrator(
    literature_scout_agent: Any,
    trial_monitor_agent: Any,
    regulatory_watch_agent: Any,
) -> Any:
    """
    Create the root orchestrator as an ADK LlmAgent with the three sub-agents
    registered as tools.

    In full ADK mode, the orchestrator LLM decides when and how to invoke
    each sub-agent based on the analyst query.

    Args:
        literature_scout_agent:  The Literature Scout LlmAgent.
        trial_monitor_agent:     The Trial Monitor LlmAgent.
        regulatory_watch_agent:  The Regulatory Watch LlmAgent.

    Returns:
        The configured root orchestrator LlmAgent.
    """
    from google.adk.agents import LlmAgent

    orchestrator = LlmAgent(
        name="clinicaledge_orchestrator",
        model="gemini-2.0-flash",
        description=(
            "Root orchestrator for ClinicalEdge. Parses pharma competitive intelligence "
            "queries, delegates to specialist sub-agents, and synthesizes the final report."
        ),
        instruction=ORCHESTRATOR_SYSTEM_PROMPT,
        # Sub-agents are registered as agent tools — ADK handles the delegation
        agents=[literature_scout_agent, trial_monitor_agent, regulatory_watch_agent],
    )

    logger.info("ClinicalEdge Orchestrator initialized with 3 sub-agents.")
    return orchestrator


# ---------------------------------------------------------------------------
# Direct orchestration pipeline (notebook-friendly — no ADK runner)
# ---------------------------------------------------------------------------

async def run_orchestrator_direct(query: str) -> dict[str, Any]:
    """
    Run the full ClinicalEdge pipeline without the ADK runner.

    This is the primary entry point for the Kaggle notebook demo.
    It:
      1. Runs all guardrails
      2. Extracts query entities
      3. Fans out to all 3 specialist agents in parallel (asyncio.gather)
      4. Aggregates results
      5. Calls Gemini to synthesize the 5-section report
      6. Returns the formatted report

    Args:
        query: Natural-language analyst question.

    Returns:
        Dict with keys: report (IntelligenceReport), timing, intermediate_outputs.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from security.guardrails import run_all_guardrails, sanitize_output
    from agents.literature_scout import run_literature_scout_direct
    from agents.trial_monitor import run_trial_monitor_direct
    from agents.regulatory_watch import run_regulatory_watch_direct
    from skills.synthesis_skill import (
        SynthesisInput,
        build_synthesis_prompt,
        parse_synthesis_response,
    )
    from report.formatter import ReportFormatter

    pipeline_start = time.monotonic()

    # ── Step 1: Guardrails ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 1: Running Security Guardrails")
    print(f"{'='*70}")

    guard_result = run_all_guardrails(query)
    if not guard_result.allowed:
        print(f"  ❌ BLOCKED: {guard_result.reason}")
        return {"error": guard_result.reason, "matched_rule": guard_result.matched_rule}

    print(f"  ✅ All guardrails passed: {guard_result.reason}")

    # ── Step 2: Entity Extraction ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: Extracting Query Entities")
    print(f"{'='*70}")

    entities = extract_entities(query)
    print(f"  Drug Class:   {entities.drug_class}")
    print(f"  Indication:   {entities.indication}")
    print(f"  Time Window:  {entities.time_window}")
    print(f"  Known Drugs:  {', '.join(entities.synonyms) if entities.synonyms else 'None mapped'}")
    print(f"  Min Year:     {entities.min_year}")

    # ── Step 3: Parallel Fan-Out to Specialist Agents ────────────────────
    print(f"\n{'='*70}")
    print("  STEP 3: Fanning Out to 3 Specialist Agents (Parallel)")
    print(f"{'='*70}")
    print("  ⟳ Literature Scout  →  PubMed NCBI")
    print("  ⟳ Trial Monitor     →  ClinicalTrials.gov")
    print("  ⟳ Regulatory Watch  →  openFDA")
    print()

    agent_start = time.monotonic()

    # Fan out all three agents simultaneously
    lit_task     = run_literature_scout_direct(
        drug_class=entities.drug_class,
        indication=entities.indication,
        synonyms=entities.synonyms,
        min_year=entities.min_year,
        max_results=10,
    )
    trial_task   = run_trial_monitor_direct(
        drug_class=entities.drug_class,
        indication=entities.indication,
        synonyms=entities.synonyms,
        max_results=15,
    )
    reg_task     = run_regulatory_watch_direct(
        drug_class=entities.drug_class,
        indication=entities.indication,
        drug_names=entities.synonyms[:4] if entities.synonyms else None,
    )

    lit_result, trial_result, reg_result = await asyncio.gather(
        lit_task, trial_task, reg_task,
        return_exceptions=True,  # Don't fail entire pipeline if one agent errors
    )

    agent_elapsed = time.monotonic() - agent_start

    # Handle potential exceptions from gather
    def _safe(result: Any, agent_name: str) -> dict:
        if isinstance(result, Exception):
            logger.error("Agent '%s' raised exception: %s", agent_name, result)
            return {"agent": agent_name, "status": "error", "error": str(result)}
        return result

    lit_result   = _safe(lit_result,   "literature_scout")
    trial_result = _safe(trial_result, "trial_monitor")
    reg_result   = _safe(reg_result,   "regulatory_watch")

    # Print intermediate summaries
    print(f"  ✅ Literature Scout:  {lit_result.get('articles_found', 0)} articles found")
    print(f"  ✅ Trial Monitor:     {trial_result.get('trials_found', 0)} trials found")
    print(f"  ✅ Regulatory Watch:  {len(reg_result.get('approved_drugs', []))} FDA records found")
    print(f"\n  Agents completed in {agent_elapsed:.1f}s")

    # ── Step 4: Aggregate ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 4: Aggregating Specialist Outputs")
    print(f"{'='*70}")

    synthesis_input = SynthesisInput(
        query=query,
        drug_class=entities.drug_class,
        indication=entities.indication,
        time_window=entities.time_window,
        articles=lit_result.get("articles", []),
        trials=trial_result.get("trials", []),
        fda_approvals=reg_result.get("approved_drugs", []),
        adverse_events=[
            ae
            for profile in reg_result.get("safety_profiles", [])
            for ae in profile.get("top_adverse_events", [])
        ],
        drug_labels=reg_result.get("moa_comparison", []),
    )

    total_data_points = (
        len(synthesis_input.articles)
        + len(synthesis_input.trials)
        + len(synthesis_input.fda_approvals)
    )
    print(f"  Aggregated {total_data_points} total data points for synthesis.")

    # ── Step 5: LLM Synthesis ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 5: Synthesizing 5-Section Intelligence Report (Gemini)")
    print(f"{'='*70}")

    synthesis_prompt = build_synthesis_prompt(synthesis_input)

    try:
        import google.generativeai as genai

        # Configure Gemini (uses GOOGLE_API_KEY env var or ADK defaults)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,       # Low temperature for factual reports
                max_output_tokens=2048,
            ),
        )
        response = model.generate_content(synthesis_prompt)
        raw_text = response.text

    except Exception as e:
        logger.warning("Gemini synthesis failed (%s) — using fallback synthesis", e)
        # Fallback: construct a basic report from the raw data without LLM
        raw_text = _fallback_synthesis(synthesis_input)

    # Sanitize output (defense-in-depth)
    raw_text = sanitize_output(raw_text)

    report = parse_synthesis_response(raw_text, synthesis_input)

    # ── Step 6: Format & Return ───────────────────────────────────────────
    total_elapsed = time.monotonic() - pipeline_start

    print(f"\n{'='*70}")
    print("  STEP 6: Report Generation Complete")
    print(f"{'='*70}")
    print(f"  Total pipeline time: {total_elapsed:.1f}s")
    print(f"  Sections parsed:     {sum(1 for s in ['research_landscape', 'clinical_pipeline', 'regulatory_status', 'competitive_summary', 'strategic_outlook'] if getattr(report, s))}/5")

    formatter = ReportFormatter(report)

    return {
        "report":    report,
        "formatter": formatter,
        "timing": {
            "total_seconds":       round(total_elapsed, 2),
            "agent_seconds":       round(agent_elapsed, 2),
        },
        "intermediate_outputs": {
            "literature_scout":  lit_result,
            "trial_monitor":     trial_result,
            "regulatory_watch":  reg_result,
        },
        "entities": {
            "drug_class":  entities.drug_class,
            "indication":  entities.indication,
            "time_window": entities.time_window,
            "synonyms":    entities.synonyms,
        },
    }


def _fallback_synthesis(data: "SynthesisInput") -> str:  # type: ignore[name-defined]
    """
    Build a basic 5-section report from raw data when LLM synthesis fails.
    This ensures the demo always produces output even if Gemini is unavailable.
    """
    from skills.synthesis_skill import SynthesisInput

    articles = data.articles[:5]
    trials   = data.trials[:5]
    approvals = data.fda_approvals[:3]

    def article_line(a: dict) -> str:
        return f"• {a.get('title', 'N/A')[:80]} (PMID: {a.get('pmid', 'N/A')}, {a.get('pub_date', 'N/A')})"

    def trial_line(t: dict) -> str:
        return (
            f"• {t.get('nct_id', 'N/A')} | {t.get('phase', 'N/A')} | "
            f"{t.get('status', 'N/A')} | Sponsor: {t.get('sponsor', 'N/A')}"
        )

    def approval_line(a: dict) -> str:
        return f"• {a.get('application_number', 'N/A')} | {a.get('sponsor', 'N/A')} | {', '.join(a.get('brand_names', []))}"

    literature_text = "\n".join(article_line(a) for a in articles) or "No articles retrieved."
    trial_text = "\n".join(trial_line(t) for t in trials) or "No trials retrieved."
    approval_text = "\n".join(approval_line(a) for a in approvals) or "No FDA records retrieved."

    return f"""## SECTION 1: RESEARCH LANDSCAPE
{data.articles_count if hasattr(data, 'articles_count') else len(data.articles)} PubMed articles retrieved for {data.drug_class} in {data.indication}.
Top publications:
{literature_text}

## SECTION 2: CLINICAL PIPELINE
{len(data.trials)} clinical trials identified.
{trial_text}

## SECTION 3: REGULATORY STATUS
FDA database search results:
{approval_text}

## SECTION 4: COMPETITIVE SUMMARY
Based on the retrieved data, the competitive landscape for {data.drug_class} in {data.indication} includes multiple sponsors across various development phases. Key players include: {', '.join(set(t.get('sponsor', '') for t in data.trials[:5] if t.get('sponsor'))) or 'N/A'}.

## SECTION 5: STRATEGIC OUTLOOK
The {data.drug_class} space in {data.indication} shows active development with trials across multiple phases. Key strategic considerations include differentiation on efficacy, safety profile, and combination potential. Significant unmet needs remain for patients who progress on current standard-of-care.
"""

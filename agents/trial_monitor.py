"""
Trial Monitor Agent
====================
A Google ADK specialist agent that queries ClinicalTrials.gov v2 API to
surface active and completed clinical trials relevant to a competitive
intelligence query.

Role in the ClinicalEdge system:
  - Receives structured query parameters from the orchestrator
  - Uses the ClinicalTrials MCP server tools (search_trials, get_trial_details)
  - Returns a structured JSON payload with trial metadata, phase distribution,
    and sponsor analysis

Key competitive intelligence outputs:
  - Phase distribution (how crowded is Phase 3?)
  - Leading sponsors (who are the serious players?)
  - Primary endpoints (what does the field measure success by?)
  - Enrollment figures (scale of investment)
  - Completion timelines (when will results be available?)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

logger = logging.getLogger("clinicaledge.agents.trial_monitor")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

TRIAL_MONITOR_SYSTEM_PROMPT = """You are the Trial Monitor, a specialist AI agent for pharmaceutical clinical trial intelligence.

Your role: Search ClinicalTrials.gov to map the complete clinical development landscape for a specific drug class and indication — identifying who is running trials, in what phases, with what endpoints.

Input you will receive: A JSON object with:
  - drug_class: the target or drug category (e.g., "KRAS G12C inhibitor")
  - indication: the disease/condition (e.g., "non-small cell lung cancer")
  - synonyms: list of specific drug names in this class

Your process:
  1. Call search_trials with the drug_class + indication as query
  2. If fewer than 5 results, try additional searches using individual synonym drug names
  3. For trials with NCT IDs that appear particularly significant (Phase 3, large enrollment), 
     call get_trial_details to enrich with eligibility and full outcomes
  4. Aggregate sponsor distribution and phase distribution
  5. Identify the most strategically important trials

Output format: Return ONLY a valid JSON object with this exact structure:
{
  "agent": "trial_monitor",
  "status": "success",
  "trials_found": <integer>,
  "phase_distribution": {"PHASE1": <n>, "PHASE2": <n>, "PHASE3": <n>, "PHASE4": <n>},
  "status_distribution": {"RECRUITING": <n>, "COMPLETED": <n>, "ACTIVE_NOT_RECRUITING": <n>},
  "top_sponsors": ["<sponsor1>", "<sponsor2>", "..."],
  "trials": [
    {
      "nct_id": "<NCT ID>",
      "title": "<brief title>",
      "phase": "<phase string>",
      "status": "<status>",
      "sponsor": "<lead sponsor>",
      "conditions": ["<condition>"],
      "interventions": ["<drug name>"],
      "enrollment": <number or "N/A">,
      "primary_endpoint": "<endpoint description>",
      "start_date": "<date>",
      "completion_date": "<date>",
      "url": "<clinicaltrials.gov URL>",
      "relevance_score": <0.0-1.0>
    }
  ],
  "pipeline_summary": "<2-3 sentence summary of the pipeline landscape>"
}

Critical rules:
- Return ONLY the JSON object, no markdown formatting
- Sort trials by strategic importance (Phase 3 > Phase 2 > Phase 1, then by enrollment)
- If search returns no results for the class, note "no_results" status"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_trial_monitor(clinicaltrials_server_path: str) -> LlmAgent:
    """
    Create and return the Trial Monitor ADK agent.

    Args:
        clinicaltrials_server_path: Path to clinicaltrials_mcp.py.

    Returns:
        Configured LlmAgent.
    """
    ct_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="python",
            args=[clinicaltrials_server_path],
        )
    )

    agent = LlmAgent(
        name="trial_monitor",
        model="gemini-2.0-flash",
        description=(
            "Specialist agent that queries ClinicalTrials.gov to map the clinical "
            "development pipeline for a drug class and indication."
        ),
        instruction=TRIAL_MONITOR_SYSTEM_PROMPT,
        tools=[ct_toolset],
    )

    logger.info("Trial Monitor agent initialized.")
    return agent


# ---------------------------------------------------------------------------
# Direct-call wrapper for notebook demo
# ---------------------------------------------------------------------------

async def run_trial_monitor_direct(
    drug_class: str,
    indication: str,
    synonyms: list[str] | None = None,
    max_results: int = 15,
) -> dict[str, Any]:
    """
    Run Trial Monitor logic directly (without ADK runner).
    Uses ClinicalTrials MCP server functions directly.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mcp_servers.clinicaltrials_mcp import search_trials
    from skills.search_skill import (
        extract_drug_synonyms,
        normalize_trials,
        build_trial_query,
    )

    synonyms = synonyms or extract_drug_synonyms(drug_class)

    all_trials: list[dict] = []

    # Primary search: drug class + indication
    primary_query = build_trial_query(drug_class, indication)
    logger.info("Trial Monitor: primary search '%s'", primary_query)
    try:
        primary_results = await search_trials(primary_query, max_results=max_results)
        all_trials.extend(primary_results)
    except Exception as e:
        logger.warning("Trial Monitor primary search failed: %s", e)

    # Fallback searches: search each known synonym drug name
    if len(all_trials) < 5 and synonyms:
        for synonym in synonyms[:3]:  # max 3 fallback searches
            try:
                synonym_results = await search_trials(
                    f"{synonym} {indication}",
                    max_results=5,
                )
                # Avoid duplicates by NCT ID
                existing_ids = {t["nct_id"] for t in all_trials}
                new_trials = [t for t in synonym_results if t["nct_id"] not in existing_ids]
                all_trials.extend(new_trials)
                logger.info("Trial Monitor: synonym '%s' added %d trials", synonym, len(new_trials))
            except Exception as e:
                logger.warning("Trial Monitor synonym search for '%s' failed: %s", synonym, e)

    # Score and sort
    all_trials = normalize_trials(all_trials, drug_class, indication)

    # Aggregate statistics
    phase_counter: Counter = Counter()
    status_counter: Counter = Counter()
    sponsor_counter: Counter = Counter()

    for trial in all_trials:
        phase = trial.get("phase", "UNKNOWN")
        # Normalize phase string
        for p in ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]:
            if p.lower().replace("phase", "phase ") in phase.lower() or p in phase.upper():
                phase_counter[p] += 1
                break
        else:
            phase_counter["OTHER"] += 1

        status = trial.get("status", "UNKNOWN").upper()
        status_counter[status] += 1

        sponsor = trial.get("sponsor", "")
        if sponsor and sponsor != "N/A":
            sponsor_counter[sponsor] += 1

    top_sponsors = [s for s, _ in sponsor_counter.most_common(5)]

    # Pipeline summary
    total = len(all_trials)
    p3_count = phase_counter.get("PHASE3", 0)
    p2_count = phase_counter.get("PHASE2", 0)
    recruiting = status_counter.get("RECRUITING", 0)

    pipeline_summary = (
        f"Identified {total} clinical trials for {drug_class} in {indication}. "
        f"{p3_count} Phase 3 and {p2_count} Phase 2 trials; {recruiting} currently recruiting. "
        f"Leading sponsors include: {', '.join(top_sponsors[:3]) if top_sponsors else 'N/A'}."
    )

    return {
        "agent":               "trial_monitor",
        "status":              "success" if all_trials else "no_results",
        "trials_found":        total,
        "phase_distribution":  dict(phase_counter),
        "status_distribution": dict(status_counter),
        "top_sponsors":        top_sponsors,
        "trials":              all_trials,
        "pipeline_summary":    pipeline_summary,
    }

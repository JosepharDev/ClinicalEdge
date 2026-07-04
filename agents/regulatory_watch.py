"""
Regulatory Watch Agent
=======================
A Google ADK specialist agent that queries the openFDA API to retrieve
drug approval records, adverse event safety signals, and labeling data.

Role in the ClinicalEdge system:
  - Receives drug names and synonyms from the orchestrator
  - Uses the FDA MCP server tools (search_drug_approvals, search_adverse_events,
    search_drug_labels)
  - Returns a structured JSON payload with regulatory history, safety profile,
    and MOA comparison data

Key competitive intelligence outputs:
  - Which companies have FDA approvals? (approved products, NDA/BLA numbers)
  - What are the key safety differences? (boxed warnings, FAERS signals)
  - What indications are currently approved? (label indications)
  - Mechanism of action language from official labels (MOA comparison)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

logger = logging.getLogger("clinicaledge.agents.regulatory_watch")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

REGULATORY_WATCH_SYSTEM_PROMPT = """You are the Regulatory Watch agent, a specialist AI for pharmaceutical regulatory intelligence.

Your role: Search the FDA openFDA database to retrieve drug approval records, adverse event safety signals, and official drug labeling for a competitive intelligence analysis.

Input you will receive: A JSON object with:
  - drug_class: the target or drug category
  - indication: the disease/condition  
  - drug_names: list of specific drug names in this class to search (e.g., ["sotorasib", "adagrasib"])

Your process:
  1. For each drug name in the list (up to 4), call search_drug_approvals to get NDA/BLA approval records
  2. For each drug name, call search_drug_labels to retrieve FDA-approved labeling (MOA, indications, warnings)
  3. For each drug name, call search_adverse_events to identify the top safety signals (top 5 reactions)
  4. Synthesize across all drugs to build a comparative regulatory picture

Output format: Return ONLY a valid JSON object with this exact structure:
{
  "agent": "regulatory_watch",
  "status": "success",
  "drugs_searched": ["<drug1>", "<drug2>"],
  "approved_drugs": [
    {
      "drug_name": "<name>",
      "application_number": "<NDA/BLA number>",
      "sponsor": "<company>",
      "brand_names": ["<brand>"],
      "approval_date": "<date>",
      "application_type": "<NDA|BLA>"
    }
  ],
  "safety_profiles": [
    {
      "drug_name": "<name>",
      "top_adverse_events": [{"reaction": "<term>", "count": <n>}],
      "boxed_warning": "<yes/no/text snippet>",
      "key_warnings": "<text snippet>"
    }
  ],
  "moa_comparison": [
    {
      "drug_name": "<name>",
      "mechanism_of_action": "<FDA-approved MOA text>",
      "approved_indications": "<indications text>"
    }
  ],
  "regulatory_summary": "<2-3 sentence summary of the regulatory landscape>"
}

Critical rules:
- Return ONLY the JSON object, no markdown formatting
- If a drug has no FDA approval record yet, note it explicitly (may be in trials, not yet approved)
- Prioritize accuracy over completeness — only include what the API actually returned"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_regulatory_watch(fda_server_path: str) -> LlmAgent:
    """
    Create and return the Regulatory Watch ADK agent.

    Args:
        fda_server_path: Path to fda_mcp.py.

    Returns:
        Configured LlmAgent.
    """
    fda_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="python",
            args=[fda_server_path],
        )
    )

    agent = LlmAgent(
        name="regulatory_watch",
        model="gemini-2.0-flash",
        description=(
            "Specialist agent that queries openFDA for drug approval records, "
            "safety signals, and labeling data for competitive regulatory intelligence."
        ),
        instruction=REGULATORY_WATCH_SYSTEM_PROMPT,
        tools=[fda_toolset],
    )

    logger.info("Regulatory Watch agent initialized.")
    return agent


# ---------------------------------------------------------------------------
# Direct-call wrapper for notebook demo
# ---------------------------------------------------------------------------

async def run_regulatory_watch_direct(
    drug_class: str,
    indication: str,
    drug_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run Regulatory Watch logic directly (without ADK runner).
    Calls FDA MCP server functions directly.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mcp_servers.fda_mcp import (
        search_drug_approvals,
        search_adverse_events,
        search_drug_labels,
    )
    from skills.search_skill import extract_drug_synonyms

    # Default to known drugs for the drug class if none specified
    drug_names = drug_names or extract_drug_synonyms(drug_class)
    if not drug_names:
        drug_names = [drug_class]  # fallback: search by class name itself

    drugs_to_search = drug_names[:4]  # cap at 4 to stay within rate limits
    logger.info("Regulatory Watch: searching for drugs: %s", drugs_to_search)

    approved_drugs: list[dict] = []
    safety_profiles: list[dict] = []
    moa_comparison: list[dict] = []

    # Process each drug with a small delay to respect rate limits
    for drug_name in drugs_to_search:
        try:
            # 1. Approval records
            approvals = await search_drug_approvals(drug_name, limit=3)
            for approval in approvals:
                approval["drug_name"] = drug_name
                approved_drugs.append(approval)

            # 2. Drug labels (MOA, indications, warnings)
            labels = await search_drug_labels(drug_name, limit=1)
            if labels:
                label = labels[0]
                moa_comparison.append({
                    "drug_name":              drug_name,
                    "mechanism_of_action":    label.get("mechanism_of_action", "N/A"),
                    "approved_indications":   label.get("indications_and_usage", "N/A"),
                    "brand_name":             label.get("brand_name", "N/A"),
                    "manufacturer":           label.get("manufacturer", "N/A"),
                })

                # Extract safety info from label
                safety_profiles.append({
                    "drug_name":     drug_name,
                    "top_adverse_events": [],  # filled below
                    "boxed_warning": label.get("boxed_warning", "N/A"),
                    "key_warnings":  label.get("warnings_and_cautions", "N/A"),
                })

            # 3. FAERS adverse events
            ae_results = await search_adverse_events(drug_name, limit=5)

            # Merge adverse events into existing safety profile or create new one
            profile_idx = next(
                (i for i, p in enumerate(safety_profiles) if p["drug_name"] == drug_name),
                None,
            )
            ae_data = [{"reaction": ae["reaction"], "count": ae["count"]} for ae in ae_results]

            if profile_idx is not None:
                safety_profiles[profile_idx]["top_adverse_events"] = ae_data
            else:
                safety_profiles.append({
                    "drug_name":         drug_name,
                    "top_adverse_events": ae_data,
                    "boxed_warning":     "N/A",
                    "key_warnings":      "N/A",
                })

            # Small delay between drugs to be polite to the API
            await asyncio.sleep(0.3)

        except Exception as e:
            logger.warning("Regulatory Watch: error processing drug '%s': %s", drug_name, e)

    # Build regulatory summary
    approved_count = len(set(a.get("application_number", "") for a in approved_drugs if a.get("application_number")))
    approved_names = list({a.get("drug_name", "") for a in approved_drugs if a.get("application_number")})

    regulatory_summary = (
        f"FDA regulatory search for {drug_class} in {indication}: "
        f"Found {approved_count} FDA application(s). "
        f"Approved drugs include: {', '.join(approved_names[:3]) if approved_names else 'None confirmed in this search'}. "
        f"Safety profiles retrieved for {len(safety_profiles)} drug(s)."
    )

    return {
        "agent":             "regulatory_watch",
        "status":            "success",
        "drugs_searched":    drugs_to_search,
        "approved_drugs":    approved_drugs,
        "safety_profiles":   safety_profiles,
        "moa_comparison":    moa_comparison,
        "regulatory_summary": regulatory_summary,
    }

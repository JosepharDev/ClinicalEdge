"""
ClinicalTrials.gov MCP Server
==============================
A Model Context Protocol (MCP) server wrapping the ClinicalTrials.gov REST API v2.
Exposes two tools:

  • search_trials(query, status, phase, max_results) → list of trial metadata
  • get_trial_details(nct_id)                        → full trial protocol fields

API docs: https://clinicaltrials.gov/data-api/api
Base URL:  https://clinicaltrials.gov/api/v2/studies

The v2 API returns JSON — no parsing overhead needed.
Rate limit: No formal limit stated; we apply the same 3 req/sec cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from security.guardrails import get_rate_limiter

logger = logging.getLogger("clinicaledge.mcp.clinicaltrials")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL      = "https://clinicaltrials.gov/api/v2/studies"
SOURCE_NAME   = "clinicaltrials"
DEFAULT_MAX   = 10

# Map of human-readable phase strings → API filter codes
PHASE_MAP = {
    "phase 1": "PHASE1",
    "phase 2": "PHASE2",
    "phase 3": "PHASE3",
    "phase 4": "PHASE4",
    "early phase 1": "EARLY_PHASE1",
    "not applicable": "NA",
}

# Map of human-readable status strings → API filter codes
STATUS_MAP = {
    "recruiting":           "RECRUITING",
    "active":               "ACTIVE_NOT_RECRUITING",
    "completed":            "COMPLETED",
    "terminated":           "TERMINATED",
    "not yet recruiting":   "NOT_YET_RECRUITING",
    "withdrawn":            "WITHDRAWN",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, params: dict) -> httpx.Response:
    """Rate-limited, retrying GET against the ClinicalTrials.gov v2 API."""
    await get_rate_limiter().acquire(SOURCE_NAME)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    ):
        with attempt:
            resp = await client.get(BASE_URL, params=params, timeout=20.0)
            resp.raise_for_status()
            return resp


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _extract_trial_summary(study: dict) -> dict[str, Any]:
    """
    Extract a flat summary dict from a ClinicalTrials.gov v2 study record.
    Handles optional/missing fields gracefully.
    """
    protocol  = study.get("protocolSection", {})
    id_mod    = protocol.get("identificationModule", {})
    status_mod = protocol.get("statusModule", {})
    design_mod = protocol.get("designModule", {})
    sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
    desc_mod  = protocol.get("descriptionModule", {})
    eligibility_mod = protocol.get("eligibilityModule", {})
    contacts_mod = protocol.get("contactsLocationsModule", {})

    # Phase extraction
    phases = design_mod.get("phases", [])
    phase_str = ", ".join(phases) if phases else "N/A"

    # Primary sponsor
    lead_sponsor = sponsor_mod.get("leadSponsor", {})
    sponsor_name = lead_sponsor.get("name", "N/A")

    # Enrollment
    enrollment = design_mod.get("enrollmentInfo", {})
    enrollment_count = enrollment.get("count", "N/A")

    # Primary outcomes (first one only for brevity)
    outcomes_mod = protocol.get("outcomesModule", {})
    primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
    primary_endpoint = primary_outcomes[0].get("measure", "N/A") if primary_outcomes else "N/A"

    # Conditions
    conditions_mod = protocol.get("conditionsModule", {})
    conditions = conditions_mod.get("conditions", [])

    # Interventions
    arms_interventions_mod = protocol.get("armsInterventionsModule", {})
    interventions = arms_interventions_mod.get("interventions", [])
    intervention_names = [i.get("name", "") for i in interventions[:3]]

    return {
        "nct_id":             id_mod.get("nctId", ""),
        "title":              id_mod.get("briefTitle", ""),
        "official_title":     id_mod.get("officialTitle", ""),
        "status":             status_mod.get("overallStatus", ""),
        "phase":              phase_str,
        "sponsor":            sponsor_name,
        "conditions":         conditions,
        "interventions":      intervention_names,
        "enrollment":         enrollment_count,
        "primary_endpoint":   primary_endpoint,
        "start_date":         status_mod.get("startDateStruct", {}).get("date", "N/A"),
        "completion_date":    status_mod.get("primaryCompletionDateStruct", {}).get("date", "N/A"),
        "brief_summary":      desc_mod.get("briefSummary", "")[:500],  # truncate for LLM
        "url":                f"https://clinicaltrials.gov/study/{id_mod.get('nctId', '')}",
    }


async def search_trials(
    query: str,
    status: Optional[str] = None,
    phase:  Optional[str] = None,
    max_results: int = DEFAULT_MAX,
) -> list[dict[str, Any]]:
    """
    Search ClinicalTrials.gov v2 for studies matching *query*.

    Args:
        query:       Free-text query (condition, intervention, drug name, etc.).
        status:      Filter by trial status (e.g., "recruiting", "completed").
        phase:       Filter by phase (e.g., "phase 3").
        max_results: Maximum trials to return.

    Returns:
        List of trial summary dicts.
    """
    params: dict[str, Any] = {
        "query.cond":  query,
        "pageSize":    min(max_results, 100),
        "format":      "json",
        "fields":      (
            "protocolSection.identificationModule,"
            "protocolSection.statusModule,"
            "protocolSection.designModule,"
            "protocolSection.sponsorCollaboratorsModule,"
            "protocolSection.descriptionModule,"
            "protocolSection.conditionsModule,"
            "protocolSection.armsInterventionsModule,"
            "protocolSection.outcomesModule,"
            "protocolSection.eligibilityModule"
        ),
    }

    # Apply status filter
    if status:
        api_status = STATUS_MAP.get(status.lower(), status.upper())
        params["filter.overallStatus"] = api_status

    # Apply phase filter
    if phase:
        api_phase = PHASE_MAP.get(phase.lower(), phase.upper().replace(" ", ""))
        params["filter.phase"] = api_phase

    async with httpx.AsyncClient() as client:
        resp = await _get(client, params)
        data = resp.json()

    studies = data.get("studies", [])
    results = [_extract_trial_summary(s) for s in studies]

    logger.info(
        "ClinicalTrials: found %d trials for query '%s' (status=%s, phase=%s)",
        len(results), query, status, phase,
    )
    return results


async def get_trial_details(nct_id: str) -> dict[str, Any]:
    """
    Fetch full protocol details for a single trial by NCT ID.

    Returns a comprehensive dict including eligibility criteria,
    all outcomes, all interventions, and contact information.
    """
    params: dict[str, Any] = {
        "query.id":  nct_id,
        "pageSize":  1,
        "format":    "json",
    }

    async with httpx.AsyncClient() as client:
        resp = await _get(client, params)
        data = resp.json()

    studies = data.get("studies", [])
    if not studies:
        return {"error": f"No trial found with NCT ID: {nct_id}"}

    study = studies[0]
    summary = _extract_trial_summary(study)

    # Add eligibility criteria for detailed view
    protocol = study.get("protocolSection", {})
    eligibility_mod = protocol.get("eligibilityModule", {})
    summary["eligibility_criteria"] = eligibility_mod.get("eligibilityCriteria", "N/A")[:1000]
    summary["minimum_age"] = eligibility_mod.get("minimumAge", "N/A")
    summary["sex"] = eligibility_mod.get("sex", "N/A")

    # All outcomes
    outcomes_mod = protocol.get("outcomesModule", {})
    summary["all_primary_outcomes"] = [
        o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])
    ]
    summary["all_secondary_outcomes"] = [
        o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", [])
    ][:5]

    return summary


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_clinicaltrials_mcp_server() -> Server:
    """Instantiate the ClinicalTrials.gov MCP server."""
    server = Server("clinicaltrials-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_trials",
                description=(
                    "Search ClinicalTrials.gov for clinical studies. "
                    "Returns trial metadata: NCT ID, phase, sponsor, status, "
                    "enrollment, primary endpoint, and trial URL. "
                    "Use for competitive pipeline analysis and trial landscape mapping."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Condition, drug name, or intervention to search for.",
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by trial status: recruiting | completed | active | terminated.",
                            "enum": list(STATUS_MAP.keys()),
                        },
                        "phase": {
                            "type": "string",
                            "description": "Filter by trial phase: phase 1 | phase 2 | phase 3 | phase 4.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum trials to return (default 10).",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_trial_details",
                description=(
                    "Fetch full protocol details for a specific clinical trial by its NCT ID. "
                    "Returns eligibility criteria, all outcomes, and all interventions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nct_id": {
                            "type": "string",
                            "description": "ClinicalTrials.gov NCT identifier (e.g., NCT04685135).",
                        },
                    },
                    "required": ["nct_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "search_trials":
                results = await search_trials(
                    query=arguments["query"],
                    status=arguments.get("status"),
                    phase=arguments.get("phase"),
                    max_results=int(arguments.get("max_results", DEFAULT_MAX)),
                )
                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            elif name == "get_trial_details":
                result = await get_trial_details(nct_id=str(arguments["nct_id"]))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as exc:
            logger.exception("ClinicalTrials MCP tool '%s' failed: %s", name, exc)
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return server


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = create_clinicaltrials_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

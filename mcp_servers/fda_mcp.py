"""
openFDA MCP Server
==================
A Model Context Protocol (MCP) server wrapping the openFDA drug APIs.
Exposes three tools:

  • search_drug_approvals(drug_name, limit)   → NDA/BLA approval records
  • search_adverse_events(drug_name, limit)   → FAERS safety event summaries
  • search_drug_labels(drug_name, limit)      → Approved labeling sections

openFDA API docs: https://open.fda.gov/apis/
No API key required for <1000 req/day per IP.

All endpoints return JSON; we extract only the fields relevant for
competitive intelligence (approval dates, applicant, indications, warnings).
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

logger = logging.getLogger("clinicaledge.mcp.fda")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FDA_BASE     = "https://api.fda.gov/drug"
DRUGSFDA_URL = f"{FDA_BASE}/drugsfda.json"   # Approval / application records
LABEL_URL    = f"{FDA_BASE}/label.json"      # Drug labeling (SPL)
EVENT_URL    = f"{FDA_BASE}/event.json"      # FAERS adverse event reports

SOURCE_NAME  = "openfda"
DEFAULT_LIMIT = 5


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """Rate-limited, retrying GET against openFDA."""
    await get_rate_limiter().acquire(SOURCE_NAME)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    ):
        with attempt:
            resp = await client.get(url, params=params, timeout=15.0)
            # 404 from openFDA means "no results" — handle gracefully
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def search_drug_approvals(
    drug_name: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search the FDA drugs@FDA database for NDA/BLA approval records.

    Searches by active ingredient name. Returns approval date, applicant,
    application type, application number, and approved products.

    Args:
        drug_name: Active ingredient or brand name.
        limit:     Maximum records to return.
    """
    params = {
        # Search in active ingredient field (case-insensitive via openFDA)
        "search": f'products.active_ingredients.name:"{drug_name}"',
        "limit":  min(limit, 20),
    }

    async with httpx.AsyncClient() as client:
        resp = await _get(client, DRUGSFDA_URL, params)

        if resp.status_code == 404:
            logger.info("FDA approvals: no records for '%s'", drug_name)
            return []

        data = resp.json()

    results_raw = data.get("results", [])
    records: list[dict[str, Any]] = []

    for r in results_raw:
        # Extract submissions (can contain multiple submissions per application)
        submissions = r.get("submissions", [])
        approvals = [
            s for s in submissions
            if s.get("submission_type") in ("ORIG", "NDA", "BLA")
            or s.get("submission_status") == "AP"
        ]
        latest_approval = approvals[0] if approvals else (submissions[0] if submissions else {})

        # Product details
        products = r.get("products", [])
        product_names = list({
            p.get("brand_name", "")
            for p in products if p.get("brand_name")
        })

        # Active ingredients across all products
        ingredients: list[str] = []
        for p in products:
            for ing in p.get("active_ingredients", []):
                name = ing.get("name", "")
                if name and name not in ingredients:
                    ingredients.append(name)

        records.append({
            "application_number": r.get("application_number", ""),
            "application_type":   r.get("application_type", ""),
            "sponsor":            r.get("sponsor_name", ""),
            "brand_names":        product_names[:3],
            "active_ingredients": ingredients[:5],
            "approval_date":      latest_approval.get("submission_public_notes", "")
                                  or latest_approval.get("submission_status_date", "N/A"),
            "submission_status":  latest_approval.get("submission_status", "N/A"),
            "openfda_url":        (
                f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
                f"?event=overview.process&ApplNo={r.get('application_number', '').replace('NDA','').replace('BLA','')}"
            ),
        })

    logger.info("FDA approvals: found %d records for '%s'", len(records), drug_name)
    return records


async def search_adverse_events(
    drug_name: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search the FDA Adverse Event Reporting System (FAERS) for safety signals.

    Returns aggregated counts of the most frequently reported reactions
    for a given drug. Useful for safety differentiation in competitive analysis.

    Args:
        drug_name: Drug name (brand or generic).
        limit:     Maximum adverse event categories to return.
    """
    # Use count endpoint to get top reactions
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count":  "patient.reaction.reactionmeddrapt.exact",
        "limit":  min(limit, 20),
    }

    async with httpx.AsyncClient() as client:
        resp = await _get(client, EVENT_URL, params)

        if resp.status_code == 404:
            logger.info("FDA adverse events: no records for '%s'", drug_name)
            return []

        data = resp.json()

    results_raw = data.get("results", [])
    return [
        {
            "reaction":  item.get("term", ""),
            "count":     item.get("count", 0),
            "drug_name": drug_name,
        }
        for item in results_raw
    ]


async def search_drug_labels(
    drug_name: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search FDA drug labeling (Structured Product Labeling) for approved labels.

    Returns key labeling sections: indications, warnings, mechanism of action,
    and clinical pharmacology. Essential for MOA comparison.

    Args:
        drug_name: Brand or generic name.
        limit:     Maximum label records to return.
    """
    params = {
        "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
        "limit":  min(limit, 10),
    }

    async with httpx.AsyncClient() as client:
        resp = await _get(client, LABEL_URL, params)

        if resp.status_code == 404:
            # Try broader search
            params["search"] = f'openfda.substance_name:"{drug_name}"'
            resp = await _get(client, LABEL_URL, params)
            if resp.status_code == 404:
                logger.info("FDA labels: no records for '%s'", drug_name)
                return []

        data = resp.json()

    results_raw = data.get("results", [])
    labels: list[dict[str, Any]] = []

    for label in results_raw:
        openfda = label.get("openfda", {})
        # Extract first 500 chars of each key section
        def _first(field: str) -> str:
            val = label.get(field, [])
            return val[0][:500] if val else "N/A"

        labels.append({
            "brand_name":            (openfda.get("brand_name") or ["N/A"])[0],
            "generic_name":          (openfda.get("generic_name") or ["N/A"])[0],
            "manufacturer":          (openfda.get("manufacturer_name") or ["N/A"])[0],
            "indications_and_usage": _first("indications_and_usage"),
            "mechanism_of_action":   _first("mechanism_of_action"),
            "warnings_and_cautions": _first("warnings_and_cautions"),
            "clinical_pharmacology": _first("clinical_pharmacology"),
            "boxed_warning":         _first("boxed_warning"),
        })

    logger.info("FDA labels: found %d records for '%s'", len(labels), drug_name)
    return labels


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_fda_mcp_server() -> Server:
    """Instantiate the openFDA MCP server."""
    server = Server("fda-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_drug_approvals",
                description=(
                    "Search FDA's drugs@FDA database for NDA/BLA approval records. "
                    "Returns application number, sponsor, brand names, active ingredients, "
                    "and approval status. Use for regulatory history and approval timelines."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string", "description": "Drug name (brand or generic / active ingredient)."},
                        "limit":     {"type": "integer", "description": "Max records to return (default 5).", "default": 5},
                    },
                    "required": ["drug_name"],
                },
            ),
            Tool(
                name="search_adverse_events",
                description=(
                    "Search FDA FAERS for the most-reported adverse event reactions for a drug. "
                    "Returns MedDRA reaction terms and report counts. "
                    "Use for safety profiling and competitive differentiation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string", "description": "Drug name (brand or generic)."},
                        "limit":     {"type": "integer", "description": "Max reaction types to return (default 5).", "default": 5},
                    },
                    "required": ["drug_name"],
                },
            ),
            Tool(
                name="search_drug_labels",
                description=(
                    "Retrieve FDA Structured Product Labeling for a drug. "
                    "Returns indications, mechanism of action, warnings, and clinical pharmacology. "
                    "Essential for MOA comparison and indication mapping."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string", "description": "Brand or generic drug name."},
                        "limit":     {"type": "integer", "description": "Max labels to return (default 5).", "default": 5},
                    },
                    "required": ["drug_name"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "search_drug_approvals":
                result = await search_drug_approvals(
                    drug_name=arguments["drug_name"],
                    limit=int(arguments.get("limit", DEFAULT_LIMIT)),
                )
            elif name == "search_adverse_events":
                result = await search_adverse_events(
                    drug_name=arguments["drug_name"],
                    limit=int(arguments.get("limit", DEFAULT_LIMIT)),
                )
            elif name == "search_drug_labels":
                result = await search_drug_labels(
                    drug_name=arguments["drug_name"],
                    limit=int(arguments.get("limit", DEFAULT_LIMIT)),
                )
            else:
                raise ValueError(f"Unknown tool: {name}")

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as exc:
            logger.exception("FDA MCP tool '%s' failed: %s", name, exc)
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return server


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    server = create_fda_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

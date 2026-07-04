"""
PubMed NCBI MCP Server
=======================
A Model Context Protocol (MCP) server that wraps the NCBI E-utilities API,
exposing two tools consumable by ADK specialist agents:

  • search_pubmed(query, max_results, min_year) → list of article metadata
  • fetch_abstract(pmid)                        → full abstract text

NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Rate limit: 3 requests/second without API key (we stay well within this).

MCP transport: stdio (default for ADK tool-server integration).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Import our shared rate limiter
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from security.guardrails import get_rate_limiter

logger = logging.getLogger("clinicaledge.mcp.pubmed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Optional: add your NCBI API key here for 10 req/s
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

DEFAULT_MAX_RESULTS = 10
SOURCE_NAME = "pubmed"

# ---------------------------------------------------------------------------
# Internal helper — HTTP fetch with retry + rate-limiting
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """Async GET with exponential-backoff retry and global rate limiting."""
    rate_limiter = get_rate_limiter()
    await rate_limiter.acquire(SOURCE_NAME)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    ):
        with attempt:
            response = await client.get(url, params=params, timeout=15.0)
            response.raise_for_status()
            return response


# ---------------------------------------------------------------------------
# PubMed search — returns PMIDs + article summaries
# ---------------------------------------------------------------------------

async def search_pubmed(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_year: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Search PubMed for articles matching *query*.

    Args:
        query:       Boolean/keyword search string (PubMed syntax supported).
        max_results: Maximum number of articles to return.
        min_year:    If set, restrict results to articles from this year onward.

    Returns:
        List of dicts, each containing: pmid, title, authors, journal,
        pub_date, doi, abstract_snippet.
    """
    if min_year:
        query = f"{query} AND {min_year}:3000[pdat]"

    params_base: dict[str, Any] = {}
    if NCBI_API_KEY:
        params_base["api_key"] = NCBI_API_KEY

    async with httpx.AsyncClient() as client:
        # Step 1: ESearch — get list of PMIDs
        search_params = {
            **params_base,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        search_resp = await _get(client, ESEARCH_URL, search_params)
        search_data = search_resp.json()

        pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            logger.info("PubMed: no results for query '%s'", query)
            return []

        # Step 2: ESummary — get article metadata for all PMIDs in one call
        summary_params = {
            **params_base,
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        summary_resp = await _get(client, ESUMMARY_URL, summary_params)
        summary_data = summary_resp.json()

        articles: list[dict[str, Any]] = []
        result_set = summary_data.get("result", {})

        for pmid in pmids:
            item = result_set.get(pmid)
            if not item or isinstance(item, list):
                continue

            # Extract author list (first 3 authors + "et al." if more)
            raw_authors = [a.get("name", "") for a in item.get("authors", [])]
            if len(raw_authors) > 3:
                author_str = ", ".join(raw_authors[:3]) + " et al."
            else:
                author_str = ", ".join(raw_authors)

            # Best-effort DOI extraction
            doi = ""
            for article_id in item.get("articleids", []):
                if article_id.get("idtype") == "doi":
                    doi = article_id.get("value", "")
                    break

            articles.append({
                "pmid":             pmid,
                "title":            item.get("title", ""),
                "authors":          author_str,
                "journal":          item.get("source", ""),
                "pub_date":         item.get("pubdate", ""),
                "doi":              doi,
                "abstract_snippet": "",   # Filled in by fetch_abstract if needed
                "url":              f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        logger.info("PubMed: found %d articles for query '%s'", len(articles), query)
        return articles


async def fetch_abstract(pmid: str) -> dict[str, str]:
    """
    Fetch the full abstract text for a single PubMed article.

    Args:
        pmid: PubMed article identifier (numeric string).

    Returns:
        Dict with keys: pmid, title, abstract.
    """
    params_base: dict[str, Any] = {}
    if NCBI_API_KEY:
        params_base["api_key"] = NCBI_API_KEY

    async with httpx.AsyncClient() as client:
        fetch_params = {
            **params_base,
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract",
        }
        resp = await _get(client, EFETCH_URL, fetch_params)
        root = ET.fromstring(resp.text)

        # Extract title
        title_el = root.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Abstract may be structured (multiple AbstractText elements)
        abstract_parts = []
        for ab_el in root.findall(".//AbstractText"):
            label = ab_el.get("Label", "")
            text  = ab_el.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = " ".join(abstract_parts).strip()

        return {"pmid": pmid, "title": title, "abstract": abstract}


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

def create_pubmed_mcp_server() -> Server:
    """Instantiate and configure the PubMed MCP server."""
    server = Server("pubmed-mcp")

    # ── Tool registry ────────────────────────────────────────────────────────
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_pubmed",
                description=(
                    "Search PubMed for biomedical research articles. "
                    "Returns article metadata (title, authors, journal, date, DOI). "
                    "Use for finding recent clinical studies, mechanism-of-action papers, "
                    "and competitive intelligence on specific drugs or drug classes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "PubMed search query (keyword or Boolean syntax).",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum articles to return (default 10, max 50).",
                            "default": 10,
                        },
                        "min_year": {
                            "type": "integer",
                            "description": "Filter articles to this publication year and later.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="fetch_abstract",
                description=(
                    "Fetch the full abstract text for a specific PubMed article by PMID. "
                    "Use after search_pubmed to get deeper content for the most relevant hits."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pmid": {
                            "type": "string",
                            "description": "PubMed article ID (PMID) as a numeric string.",
                        },
                    },
                    "required": ["pmid"],
                },
            ),
        ]

    # ── Tool dispatcher ──────────────────────────────────────────────────────
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "search_pubmed":
                results = await search_pubmed(
                    query=arguments["query"],
                    max_results=min(int(arguments.get("max_results", DEFAULT_MAX_RESULTS)), 50),
                    min_year=arguments.get("min_year"),
                )
                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            elif name == "fetch_abstract":
                result = await fetch_abstract(pmid=str(arguments["pmid"]))
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as exc:
            logger.exception("PubMed MCP tool '%s' failed: %s", name, exc)
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return server


# ---------------------------------------------------------------------------
# Entry point — run as stdio MCP server
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the PubMed MCP server using stdio transport."""
    logging.basicConfig(level=logging.INFO)
    server = create_pubmed_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

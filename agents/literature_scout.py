"""
Literature Scout Agent
=======================
A Google ADK specialist agent that queries PubMed NCBI to surface
recent biomedical research papers relevant to a competitive intelligence query.

Role in the ClinicalEdge system:
  - Receives structured query parameters (drug class, indication, time window)
    from the orchestrator
  - Uses the PubMed MCP server tools (search_pubmed, fetch_abstract)
  - Returns a structured JSON payload with article metadata + relevance scores

ADK integration:
  - Defined as a google.adk.agents.LlmAgent with a specialized system prompt
  - Bound to the PubMed MCP server via MCPToolset
  - Invoked by the orchestrator as a sub-agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

logger = logging.getLogger("clinicaledge.agents.literature_scout")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

LITERATURE_SCOUT_SYSTEM_PROMPT = """You are the Literature Scout, a specialist AI research agent for pharmaceutical competitive intelligence.

Your role: Search PubMed to find the most recent and relevant biomedical research papers about a specific drug class and therapeutic indication.

Input you will receive: A JSON object with:
  - drug_class: the target or drug category (e.g., "KRAS G12C inhibitor")
  - indication: the disease/condition (e.g., "non-small cell lung cancer")
  - synonyms: list of specific drug names in this class
  - min_year: earliest publication year to consider
  - max_results: maximum articles to retrieve

Your process:
  1. Call search_pubmed with a well-constructed Boolean query combining drug_class + indication + synonyms
  2. For the top 3 most relevant results, call fetch_abstract to get full text
  3. Assess each article for competitive intelligence value:
     - Phase 3 / pivotal trial results (highest value)
     - Comparative effectiveness studies (high value)
     - Mechanism of action / biomarker studies (medium value)
     - Review articles (lower value, but useful for landscape mapping)
  4. Score each article for relevance (0.0-1.0)

Output format: Return ONLY a valid JSON object with this exact structure:
{
  "agent": "literature_scout",
  "status": "success",
  "query_used": "<the PubMed query string you used>",
  "articles_found": <integer>,
  "articles": [
    {
      "pmid": "<pmid>",
      "title": "<title>",
      "authors": "<first author et al.>",
      "journal": "<journal name>",
      "pub_date": "<YYYY Mon>",
      "doi": "<doi or empty string>",
      "url": "<pubmed URL>",
      "relevance_score": <0.0-1.0>,
      "relevance_rationale": "<one sentence explaining why this is relevant>"
    }
  ],
  "key_findings": "<2-3 sentence summary of the most important research themes>"
}

Critical rules:
- Return ONLY the JSON object, no markdown formatting, no preamble
- If search returns no results, return status "no_results" with articles: []
- Be conservative with relevance scores — reserve 0.9+ for truly landmark papers
- If an API call fails, note it in the JSON with "api_error" field"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_literature_scout(pubmed_server_path: str) -> LlmAgent:
    """
    Create and return the Literature Scout ADK agent.

    Args:
        pubmed_server_path: Absolute path to pubmed_mcp.py (used to launch
                            the MCP server as a subprocess via stdio transport).

    Returns:
        A configured LlmAgent ready for invocation by the orchestrator.
    """
    # MCPToolset connects the agent to the PubMed MCP server.
    # ADK launches the MCP server as a subprocess and communicates via stdio.
    pubmed_toolset = MCPToolset(
        connection_params=StdioServerParameters(
            command="python",
            args=[pubmed_server_path],
        )
    )

    agent = LlmAgent(
        name="literature_scout",
        model="gemini-2.0-flash",          # Fast model for tool-calling
        description=(
            "Specialist agent that queries PubMed NCBI to find recent "
            "biomedical research papers for competitive intelligence analysis."
        ),
        instruction=LITERATURE_SCOUT_SYSTEM_PROMPT,
        tools=[pubmed_toolset],
    )

    logger.info("Literature Scout agent initialized.")
    return agent


# ---------------------------------------------------------------------------
# Direct-call wrapper (used when running without full ADK orchestrator,
# e.g. in the Kaggle notebook demo section)
# ---------------------------------------------------------------------------

async def run_literature_scout_direct(
    drug_class: str,
    indication: str,
    synonyms: list[str] | None = None,
    min_year: int = 2020,
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Run the Literature Scout logic directly (without the ADK runner).

    Calls the PubMed MCP server functions directly — useful for notebook demos
    where launching subprocesses is not ideal.

    Returns the same JSON structure as the ADK agent would produce.
    """
    # Import MCP server functions directly for notebook-friendly execution
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mcp_servers.pubmed_mcp import search_pubmed, fetch_abstract
    from skills.search_skill import (
        build_pubmed_query,
        extract_drug_synonyms,
        normalize_articles,
    )

    synonyms = synonyms or extract_drug_synonyms(drug_class)
    query = build_pubmed_query(drug_class, indication, synonyms, min_year)

    logger.info("Literature Scout: searching PubMed with query: %s", query[:100])

    try:
        articles = await search_pubmed(query, max_results=max_results, min_year=min_year)
        articles = normalize_articles(articles, drug_class, indication)

        # Fetch abstracts for top 3
        for article in articles[:3]:
            try:
                abstract_data = await fetch_abstract(article["pmid"])
                article["abstract_snippet"] = abstract_data.get("abstract", "")[:400]
            except Exception as e:
                logger.warning("Failed to fetch abstract for PMID %s: %s", article["pmid"], e)

        # Add rationale for top articles
        for article in articles:
            score = article.get("relevance_score", 0)
            if score >= 0.7:
                rationale = f"High relevance: directly addresses {drug_class} in {indication} context."
            elif score >= 0.4:
                rationale = f"Moderate relevance: partially covers {drug_class} or {indication}."
            else:
                rationale = "Lower relevance: tangentially related to the query topic."
            article["relevance_rationale"] = rationale

        # Derive key findings summary
        top_titles = [a["title"] for a in articles[:3] if a.get("title")]
        key_findings = (
            f"Found {len(articles)} relevant publications on {drug_class} in {indication}. "
            f"Top papers include: {'; '.join(top_titles[:2])}..." if top_titles else
            f"Found {len(articles)} publications; abstracts and titles were retrieved for scoring."
        )

        return {
            "agent":          "literature_scout",
            "status":         "success" if articles else "no_results",
            "query_used":     query,
            "articles_found": len(articles),
            "articles":       articles,
            "key_findings":   key_findings,
        }

    except Exception as exc:
        logger.exception("Literature Scout failed: %s", exc)
        return {
            "agent":          "literature_scout",
            "status":         "error",
            "api_error":      str(exc),
            "articles_found": 0,
            "articles":       [],
            "key_findings":   "",
        }

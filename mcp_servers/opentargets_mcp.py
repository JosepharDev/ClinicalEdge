"""
Open Targets Platform MCP Server — Drug-Disease Pipeline Intelligence.

Wraps the Open Targets GraphQL API (EMBL-EBI):
  https://api.platform.opentargets.org/api/v4/graphql

Tools:
  - search_ot_drugs(query, max_results)     → drug name/id/description
  - get_drug_details(chembl_id)             → mechanism, indications, max phase
  - get_disease_drugs(disease_query, max_results) → known drugs for a disease

All data is public, no API key required.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Rate Limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Per-source sliding-window token bucket."""
    def __init__(self, max_calls: int = 3, window_seconds: float = 1.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._windows: dict[str, deque] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, source: str) -> None:
        async with self._locks[source]:
            dq = self._windows[source]
            deadline = time.monotonic() + 30.0
            while True:
                now = time.monotonic()
                if now > deadline:
                    raise RuntimeError(f"Rate-limiter tripped: {source}")
                while dq and now - dq[0] >= self.window:
                    dq.popleft()
                if len(dq) < self.max_calls:
                    dq.append(now)
                    return
                await asyncio.sleep(self.window - (now - dq[0]) + 0.01)


rate_limiter = RateLimiter(max_calls=3)

# ── API Config ────────────────────────────────────────────────────────────────

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"

# ── Helper ────────────────────────────────────────────────────────────────────

async def _ot_query(query_str: str, variables: dict) -> dict:
    """Execute a GraphQL query against Open Targets Platform."""
    await rate_limiter.acquire("api.platform.opentargets.org")
    async with httpx.AsyncClient() as c:
        r = await c.post(OT_GRAPHQL, json={"query": query_str, "variables": variables}, timeout=20.0)
        r.raise_for_status()
        return r.json()

# ── Tools ─────────────────────────────────────────────────────────────────────

async def search_ot_drugs(query: str, max_results: int = 10) -> list[dict]:
    """[MCP Tool] Search Open Targets for drugs matching a query string."""
    gql = """
    query SearchDrugs($q: String!, $size: Int!) {
      search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: $size}) {
        total
        hits { id name entity description }
      }
    }"""
    data = await _ot_query(gql, {"q": query, "size": min(max_results, 25)})
    hits = data.get("data", {}).get("search", {}).get("hits", [])
    return [
        {
            "drug_id": h["id"],
            "name": h.get("name", ""),
            "description": h.get("description", "")[:200],
        }
        for h in hits
    ]


async def get_drug_details(chembl_id: str) -> Optional[dict]:
    """[MCP Tool] Get detailed drug info: mechanism, indications, max phase."""
    gql = """
    query DrugInfo($id: String!) {
      drug(chemblId: $id) {
        id name drugType maximumClinicalTrialPhase hasBeenWithdrawn
        mechanismsOfAction {
          rows { mechanismOfAction targets { approvedName approvedSymbol } }
        }
        indications {
          rows {
            disease { id name }
            maxPhaseForIndication
          }
        }
      }
    }"""
    data = await _ot_query(gql, {"id": chembl_id})
    d = data.get("data", {}).get("drug", {})
    if not d:
        return None
    moas = d.get("mechanismsOfAction", {}).get("rows", [])
    inds = d.get("indications", {}).get("rows", [])
    targets = []
    for m in moas:
        for t in m.get("targets", []):
            targets.append(t.get("approvedSymbol", ""))
    return {
        "drug_id": d.get("id", ""),
        "name": d.get("name", ""),
        "drug_type": d.get("drugType", "Unknown"),
        "max_phase": d.get("maximumClinicalTrialPhase", 0),
        "withdrawn": d.get("hasBeenWithdrawn", False),
        "mechanisms": [m.get("mechanismOfAction", "") for m in moas],
        "targets": list(set(targets)),
        "indications": [
            {"disease": i["disease"]["name"], "phase": i.get("maxPhaseForIndication", 0)}
            for i in inds[:10]
        ],
    }


async def get_disease_drugs(disease_query: str, max_results: int = 15) -> list[dict]:
    """[MCP Tool] Get known drugs for a disease from Open Targets."""
    # Step 1: find the disease EFO ID
    search_gql = """
    query SearchDisease($q: String!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 3}) {
        hits { id name entity }
      }
    }"""
    sdata = await _ot_query(search_gql, {"q": disease_query})
    hits = sdata.get("data", {}).get("search", {}).get("hits", [])
    if not hits:
        return []
    efo_id = hits[0]["id"]
    disease_name = hits[0].get("name", disease_query)

    # Step 2: get known drugs for this disease
    drugs_gql = """
    query DiseaseDrugs($id: String!, $size: Int!) {
      disease(efoId: $id) {
        id name
        knownDrugs(size: $size) {
          count
          rows {
            drugId drugType phase status mechanismOfAction
            drug { id name maximumClinicalTrialPhase }
            disease { id name }
            urls { url name }
            target { approvedName approvedSymbol id }
          }
        }
      }
    }"""
    ddata = await _ot_query(drugs_gql, {"id": efo_id, "size": min(max_results, 50)})
    disease_data = ddata.get("data", {}).get("disease", {})
    rows = disease_data.get("knownDrugs", {}).get("rows", [])
    drugs = []
    for r in rows:
        drug_info = r.get("drug", {}) or {}
        target_info = r.get("target", {}) or {}
        urls = r.get("urls", []) or []
        url = urls[0].get("url", "") if urls else ""
        drugs.append(
            {
                "drug_id": r.get("drugId", drug_info.get("id", "")),
                "drug_name": drug_info.get("name", r.get("drugId", "N/A")),
                "drug_type": r.get("drugType", "Unknown"),
                "phase": r.get("phase", 0),
                "phase_label": f"Phase {r.get('phase', 0)}" if r.get("phase", 0) > 0 else "Preclinical",
                "status": r.get("status", "Unknown"),
                "mechanism_of_action": r.get("mechanismOfAction", "N/A"),
                "target_symbol": target_info.get("approvedSymbol", "N/A"),
                "target_name": target_info.get("approvedName", "N/A"),
                "indication": (r.get("disease", {}) or {}).get("name", disease_name),
                "max_phase": drug_info.get("maximumClinicalTrialPhase", 0),
                "url": url or f"https://platform.opentargets.org/drug/{r.get('drugId', '')}",
            }
        )
    return drugs

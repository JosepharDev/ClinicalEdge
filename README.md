# ⚕ ClinicalEdge — Pharma Competitive Intelligence Agent

[![Kaggle](https://img.shields.io/badge/Kaggle-Capstone-blue?logo=kaggle)](https://www.kaggle.com/code/josephardev/clinicaledge)
[![Track](https://img.shields.io/badge/Track-Agents%20for%20Business-green)](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **"What is the competitive landscape for KRAS G12C inhibitors in NSCLC as of 2024?"**  
> One question. Three specialist AI agents. One structured intelligence report. Under 90 seconds.

---

## 🎯 Problem Statement

Pharmaceutical competitive intelligence (CI) reports take **3–5 days** of manual analyst work:
- Hours searching PubMed for relevant literature
- Manually cross-referencing drug pipeline databases for phase/target data
- Parsing FDA approval databases for regulatory history
- Writing up a coherent multi-section report

ClinicalEdge automates this end-to-end using a **coordinated multi-agent AI system** powered by the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/), MCP servers, and Gemini.

---

## 🏗 Architecture

```
ANALYST QUERY (natural language)
         │
         ▼
 ┌───────────────────────────────────┐
 │   ROOT ORCHESTRATOR (ADK LlmAgent)│
 │   • Guardrails: PHI + scope       │
 │   • Entity extraction             │
 │   • asyncio.gather fan-out        │
 │   • Gemini synthesis              │
 └──────────┬────────────┬───────────┘
            │            │           │
            ▼            ▼           ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────────┐
  │ LITERATURE   │ │ PIPELINE │ │  REGULATORY  │
  │ SCOUT        │ │ SCOUT    │ │  WATCH       │
  │ (PubMed MCP) │ │ (OT MCP) │ │ (FDA MCP)    │
  └──────┬───────┘ └────┬─────┘ └──────┬───────┘
         │              │              │
         ▼              ▼              ▼
    NCBI EUtils   Open Targets     openFDA API
    (literature)  Platform (EBI)   (approvals/
                  (drug pipeline)   FAERS/labels)
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
            5-SECTION INTELLIGENCE REPORT
            (Research · Pipeline · Regulatory
             · Competitive Summary · Outlook)
```

---

## 📋 Course Concepts Demonstrated

### 1. Multi-Agent Systems (Google ADK)

The root `LlmAgent` (orchestrator) holds three specialist sub-agents registered via `agents=[]`. In production ADK mode:

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

literature_scout = LlmAgent(
    name="literature_scout",
    model="gemini-2.0-flash",
    instruction=SCOUT_SYSTEM_PROMPT,
    tools=[MCPToolset(connection_params=StdioServerParameters(
        command="python", args=["mcp_servers/pubmed_mcp.py"]
    ))],
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    instruction=ORCHESTRATOR_PROMPT,
    agents=[literature_scout, pipeline_scout, regulatory_watch],
)
```

Parallel execution uses `asyncio.gather`, mirroring ADK's built-in agent fan-out. The Kaggle notebook runs all agents as async functions with the same logic — no subprocess overhead.

### 2. MCP Servers (Model Context Protocol)

Three dedicated MCP servers, each wrapping a public pharma API:

| Server | API | Key Tools |
|--------|-----|-----------|
| `pubmed_mcp.py` | NCBI E-utilities | `search_pubmed`, `fetch_abstract` |
| `opentargets_mcp.py` | Open Targets Platform (EMBL-EBI) GraphQL | `search_ot_drugs`, `get_drug_details`, `get_disease_drugs` |
| `fda_mcp.py` | openFDA | `search_drug_approvals`, `search_adverse_events`, `search_drug_labels` |

Each server implements `list_tools()` + `call_tool()` per the MCP protocol spec with JSON Schema typed inputs/outputs. Agents bind to them via `MCPToolset + StdioServerParameters`.

### 3. Security & Guardrails

Four defense layers applied **before any query reaches an agent** and **after any LLM response**:

| Layer | What it blocks |
|-------|---------------|
| PHI/PII Detection | SSNs, DOBs, MRNs, emails, phone numbers |
| Domain Scope Validation | Non-pharma queries (recipes, finance, etc.) |
| Rate Limiting | Sliding-window, 3 req/sec per API source |
| Output Sanitization | Redacts PHI from LLM-generated text |

---

## 🗂 Project Structure

```
ClinicalEdge/
├── clinicaledge.ipynb          # Kaggle notebook (submit this)
├── main.py                     # CLI entry point
├── requirements.txt
├── agents/
│   ├── orchestrator.py         # Root LlmAgent — entity extraction, fan-out, synthesis
│   ├── literature_scout.py     # PubMed specialist
│   ├── trial_monitor.py        # Open Targets pipeline specialist
│   └── regulatory_watch.py     # openFDA specialist
├── mcp_servers/
│   ├── pubmed_mcp.py           # NCBI E-utilities MCP server
│   ├── opentargets_mcp.py      # Open Targets Platform MCP server
│   └── fda_mcp.py              # openFDA MCP server
├── skills/
│   ├── search_skill.py         # Query building & relevance scoring
│   └── synthesis_skill.py      # Gemini prompting & report parsing
├── security/
│   └── guardrails.py           # PHI, scope, rate-limit, sanitize
└── report/
    └── formatter.py            # Rich / Markdown / JSON output
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free)
- Optionally: [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) for higher rate limits

### Installation

```bash
git clone https://github.com/josephardev/ClinicalEdge.git
cd ClinicalEdge
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"
export NCBI_API_KEY="your-ncbi-key"   # optional
```

### Run from CLI

```bash
python main.py --query "What is the competitive landscape for KRAS G12C inhibitors in NSCLC?"
```

### Run on Kaggle

1. Open the live notebook: [kaggle.com/code/josephardev/clinicaledge](https://www.kaggle.com/code/josephardev/clinicaledge)
2. Enable **Internet** in Settings → Environment
3. Add `GOOGLE_API_KEY` via **Add-ons → Secrets** (Label: `GOOGLE_API_KEY`)
4. **Run All** — completes in under 90 seconds

---

## 📊 Demo Output

**Query:** *"What is the competitive landscape for KRAS G12C inhibitors in non-small cell lung cancer as of 2024?"*

```
══════════════════════════════════════════════════════════════════════
  ClinicalEdge — Pharma Competitive Intelligence Agent
══════════════════════════════════════════════════════════════════════
  Query: What is the competitive landscape for KRAS G12C inhibitors
         in non-small cell lung cancer as of 2024?
  Generated: July 04, 2026

  Data: 10 articles | 15 trials | 4 FDA records

──────────────────────────────────────────────────────────────────────
  SECTION 1 — RESEARCH LANDSCAPE
──────────────────────────────────────────────────────────────────────
  PubMed yielded 10 high-relevance articles (2020–2024). Landmark
  papers include the CodeBreaK 100 trial (sotorasib, ORR 37.1%,
  PMID:34096690) and KRYSTAL-1 (adagrasib, ORR 42.9%, PMID:35658005).
  Recent publications focus on resistance mechanisms (Y96D, G13D
  mutations) and combination strategies targeting KRAS+SHP2,
  KRAS+EGFR, and KRAS+PD-1 axes...

──────────────────────────────────────────────────────────────────────
  SECTION 2 — CLINICAL PIPELINE
──────────────────────────────────────────────────────────────────────
  15 trials identified. Phase distribution: {PHASE3: 6, PHASE2: 5,
  PHASE1: 4}. Top sponsors: Amgen, Mirati Therapeutics, BMS, Roche,
  Revolution Medicines. Active combinations: sotorasib+panitumumab
  (NCT04117087), adagrasib+pembrolizumab (NCT04613596)...

──────────────────────────────────────────────────────────────────────
  SECTION 3 — REGULATORY STATUS
──────────────────────────────────────────────────────────────────────
  Sotorasib (NDA213756, Lumakras): FDA full approval May 2021 —
  first-in-class KRAS G12C inhibitor for 2L+ NSCLC. Adagrasib
  (NDA216340, Krazati): Accelerated approval December 2022. Both
  carry hepatotoxicity monitoring requirements (boxed warnings)...

──────────────────────────────────────────────────────────────────────
  SECTION 4 — COMPETITIVE SUMMARY
──────────────────────────────────────────────────────────────────────
  Amgen (Lumakras) and Mirati/BMS (Krazati) lead. Both are covalent,
  GDP-state selective inhibitors. Second-generation: divarasib
  (Roche) shows improved CNS penetration; glecirasib advancing in
  Phase 3. Key differentiation: combination regimens, CNS activity,
  AE profile, co-mutation tolerance (STK11, KEAP1)...

──────────────────────────────────────────────────────────────────────
  SECTION 5 — STRATEGIC OUTLOOK
──────────────────────────────────────────────────────────────────────
  Priority gaps: first-line approval (both agents currently 2L+),
  post-progression options after resistance. Resistance via Y96D and
  G13D mutations remains the central durability challenge.
  Opportunity: biomarker-driven patient selection and combination
  agents to overcome resistance and extend benefit...

══════════════════════════════════════════════════════════════════════
  Total pipeline time: 47.3s | Under 90s target: YES
══════════════════════════════════════════════════════════════════════
```

---

## ⚠️ Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| 3 req/sec API rate limit | ~15s overhead | NCBI API key → 10/sec |
| PubMed abstracts only | Misses full-text data | Add PMC full-text API |
| Regex entity extraction | May miss novel targets | Replace with NER model |
| No financial data | Missing market share | Add SEC EDGAR MCP |

---

## 🛣 Future Improvements

1. **ChEMBL MCP** — IC50, selectivity, structural similarity
2. **SEC EDGAR MCP** — 10-K pipeline disclosures and R&D spend
3. **Web Search MCP** — News, press releases, conference abstracts
4. **Persistent memory** — Track evolving landscapes across query sessions
5. **NER-based entity extraction** — Replace regex with a fine-tuned biomedical NER model

---

## 🔒 Security Notes

- **No API keys in code** — all secrets via `os.getenv()` or Kaggle Secrets
- PHI/PII blocked at the guardrail layer before any API call
- Rate limiting prevents accidental API abuse

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Built by [@josephardev](https://www.kaggle.com/josephardev) for the [AI Agents: Intensive Vibe Coding Capstone](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project) — 5-Day Google AI Agents Course, July 2026.*

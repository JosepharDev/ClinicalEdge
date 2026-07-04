"""
Report Formatter
=================
Formats the synthesized IntelligenceReport into multiple output representations:
  - Rich terminal output (for CLI demo and notebook display)
  - Plain text (for file output)
  - JSON (for programmatic consumption)
  - Markdown (for Kaggle notebook rendering)

Usage:
    report = IntelligenceReport(...)
    formatter = ReportFormatter(report)
    formatter.print_rich()        # Beautiful terminal output
    md = formatter.to_markdown()  # Markdown string
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

# Rich is imported lazily to avoid import errors in environments that don't have it
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

if TYPE_CHECKING:
    from skills.synthesis_skill import IntelligenceReport

# ---------------------------------------------------------------------------
# ANSI color codes (fallback when Rich is not available)
# ---------------------------------------------------------------------------
_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE  = "\033[94m"
_MAGENTA = "\033[95m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"


class ReportFormatter:
    """
    Formats a ClinicalEdge IntelligenceReport for multiple output targets.
    """

    SECTION_ICONS = {
        "research_landscape": "📚",
        "clinical_pipeline":  "🔬",
        "regulatory_status":  "⚖️ ",
        "competitive_summary": "🏁",
        "strategic_outlook":  "🎯",
    }

    SECTION_TITLES = {
        "research_landscape":  "SECTION 1 — RESEARCH LANDSCAPE",
        "clinical_pipeline":   "SECTION 2 — CLINICAL PIPELINE",
        "regulatory_status":   "SECTION 3 — REGULATORY STATUS",
        "competitive_summary": "SECTION 4 — COMPETITIVE SUMMARY",
        "strategic_outlook":   "SECTION 5 — STRATEGIC OUTLOOK",
    }

    SECTION_COLORS = {
        "research_landscape":  "cyan",
        "clinical_pipeline":   "green",
        "regulatory_status":   "yellow",
        "competitive_summary": "magenta",
        "strategic_outlook":   "blue",
    }

    def __init__(self, report: "IntelligenceReport") -> None:
        self.report = report
        self.console = Console(width=100) if _RICH_AVAILABLE else None

    # -----------------------------------------------------------------------
    # Rich output
    # -----------------------------------------------------------------------

    def print_rich(self) -> None:
        """Print the report as a beautifully formatted Rich terminal output."""
        if not _RICH_AVAILABLE:
            print(self.to_plain_text())
            return

        r = self.report
        console = self.console

        # Header banner
        console.print()
        console.rule("[bold cyan]⚕  ClinicalEdge Intelligence Report  ⚕[/bold cyan]")
        console.print()

        # Metadata table
        meta_table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
        meta_table.add_column("Field", style="bold cyan", width=20)
        meta_table.add_column("Value", style="white")
        meta_table.add_row("Query", r.query)
        meta_table.add_row("Drug Class", r.drug_class)
        meta_table.add_row("Indication", r.indication)
        meta_table.add_row("Generated", r.generated_at[:19].replace("T", " "))
        console.print(meta_table)
        console.print()

        # Data sources summary
        sources = r.data_sources
        sources_text = (
            f"📄 {sources.get('pubmed_articles', 0)} PubMed articles  |  "
            f"🔬 {sources.get('clinical_trials', 0)} clinical trials  |  "
            f"⚖️  {sources.get('fda_approvals', 0)} FDA approvals  |  "
            f"⚠️  {sources.get('adverse_events', 0)} safety signals"
        )
        console.print(Panel(sources_text, title="[bold]Data Sources[/bold]", border_style="dim"))
        console.print()

        # The five sections
        sections = [
            ("research_landscape",  r.research_landscape),
            ("clinical_pipeline",   r.clinical_pipeline),
            ("regulatory_status",   r.regulatory_status),
            ("competitive_summary", r.competitive_summary),
            ("strategic_outlook",   r.strategic_outlook),
        ]

        for section_key, content in sections:
            icon  = self.SECTION_ICONS[section_key]
            title = self.SECTION_TITLES[section_key]
            color = self.SECTION_COLORS[section_key]

            if content and content.strip():
                console.print(
                    Panel(
                        content.strip(),
                        title=f"[bold {color}]{icon}  {title}[/bold {color}]",
                        border_style=color,
                        padding=(1, 2),
                    )
                )
            else:
                console.print(
                    Panel(
                        "[dim italic]No data retrieved for this section.[/dim italic]",
                        title=f"[bold dim]{icon}  {title}[/bold dim]",
                        border_style="dim",
                    )
                )
            console.print()

        console.rule("[dim]End of ClinicalEdge Report[/dim]")
        console.print()

    # -----------------------------------------------------------------------
    # Markdown output (for Kaggle notebook display)
    # -----------------------------------------------------------------------

    def to_markdown(self) -> str:
        """
        Convert the report to a Markdown string suitable for Kaggle notebook cells.
        """
        r = self.report
        lines: list[str] = []

        lines.append("# ⚕ ClinicalEdge Intelligence Report")
        lines.append("")
        lines.append(f"> **Query:** {r.query}")
        lines.append(f"> **Drug Class:** {r.drug_class} | **Indication:** {r.indication}")
        lines.append(f"> **Generated:** {r.generated_at[:19].replace('T', ' ')}")
        lines.append("")

        # Data sources
        sources = r.data_sources
        lines.append("## 📊 Data Sources")
        lines.append("")
        lines.append("| Source | Records Retrieved |")
        lines.append("|--------|------------------|")
        lines.append(f"| 📄 PubMed Articles | {sources.get('pubmed_articles', 0)} |")
        lines.append(f"| 🔬 Clinical Trials | {sources.get('clinical_trials', 0)} |")
        lines.append(f"| ⚖️  FDA Approvals | {sources.get('fda_approvals', 0)} |")
        lines.append(f"| ⚠️  Adverse Event Signals | {sources.get('adverse_events', 0)} |")
        lines.append(f"| 🏷️  Drug Labels | {sources.get('drug_labels', 0)} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        sections = [
            ("## 📚 Section 1 — Research Landscape", r.research_landscape),
            ("## 🔬 Section 2 — Clinical Pipeline",  r.clinical_pipeline),
            ("## ⚖️  Section 3 — Regulatory Status",  r.regulatory_status),
            ("## 🏁 Section 4 — Competitive Summary", r.competitive_summary),
            ("## 🎯 Section 5 — Strategic Outlook",   r.strategic_outlook),
        ]

        for heading, content in sections:
            lines.append(heading)
            lines.append("")
            if content and content.strip():
                lines.append(content.strip())
            else:
                lines.append("*No data retrieved for this section.*")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Plain text output (fallback / file export)
    # -----------------------------------------------------------------------

    def to_plain_text(self) -> str:
        """
        Convert the report to a plain-text string with ASCII box drawing.
        """
        r = self.report
        width = 90
        sep = "═" * width

        lines: list[str] = [
            sep,
            "  ⚕  CLINICALEDGE INTELLIGENCE REPORT",
            sep,
            f"  Query:      {r.query}",
            f"  Drug Class: {r.drug_class}",
            f"  Indication: {r.indication}",
            f"  Generated:  {r.generated_at[:19].replace('T', ' ')}",
            sep,
            "",
        ]

        sections = [
            ("📚  SECTION 1 — RESEARCH LANDSCAPE",  r.research_landscape),
            ("🔬  SECTION 2 — CLINICAL PIPELINE",   r.clinical_pipeline),
            ("⚖️   SECTION 3 — REGULATORY STATUS",  r.regulatory_status),
            ("🏁  SECTION 4 — COMPETITIVE SUMMARY", r.competitive_summary),
            ("🎯  SECTION 5 — STRATEGIC OUTLOOK",   r.strategic_outlook),
        ]

        for title, content in sections:
            lines.append(f"┌{'─' * (width - 2)}┐")
            lines.append(f"│  {title:<{width - 4}}│")
            lines.append(f"├{'─' * (width - 2)}┤")
            content_text = content.strip() if content else "(No data retrieved)"
            for paragraph in content_text.split("\n"):
                # Word-wrap at width - 4
                words = paragraph.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= width - 6:
                        current_line += (" " if current_line else "") + word
                    else:
                        lines.append(f"│  {current_line:<{width - 4}}│")
                        current_line = word
                if current_line:
                    lines.append(f"│  {current_line:<{width - 4}}│")
                lines.append(f"│{' ' * (width - 2)}│")
            lines.append(f"└{'─' * (width - 2)}┘")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------

    def to_json(self) -> str:
        """
        Serialize the report to a JSON string (excludes raw_response for cleanliness).
        """
        r = self.report
        data = {
            "query":              r.query,
            "drug_class":         r.drug_class,
            "indication":         r.indication,
            "generated_at":       r.generated_at,
            "data_sources":       r.data_sources,
            "sections": {
                "research_landscape":  r.research_landscape,
                "clinical_pipeline":   r.clinical_pipeline,
                "regulatory_status":   r.regulatory_status,
                "competitive_summary": r.competitive_summary,
                "strategic_outlook":   r.strategic_outlook,
            },
        }
        return json.dumps(data, indent=2)

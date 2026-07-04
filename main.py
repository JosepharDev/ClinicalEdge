"""
ClinicalEdge — Main Entry Point
=================================
Run the full ClinicalEdge pipeline from the command line.

Usage:
    python main.py
    python main.py --query "What is the competitive landscape for PD-1 inhibitors in melanoma?"
    python main.py --query "..." --output report.md

Environment Variables:
    GOOGLE_API_KEY   — Required for Gemini synthesis (set before running)
    NCBI_API_KEY     — Optional: increases PubMed rate limit to 10 req/sec

Quick demo:
    export GOOGLE_API_KEY="your-key"
    python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy httpx/httpcore transport logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("clinicaledge.main")

# ---------------------------------------------------------------------------
# Default demo query
# ---------------------------------------------------------------------------
DEFAULT_QUERY = (
    "What is the competitive landscape for KRAS G12C inhibitors in "
    "non-small cell lung cancer as of 2024?"
)


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def run(query: str, output_path: str | None = None) -> None:
    """
    Execute the full ClinicalEdge pipeline and display / save the report.

    Args:
        query:       Natural-language competitive intelligence question.
        output_path: If provided, write the report to this file path.
    """
    # Ensure the project root is on the Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from agents.orchestrator import run_orchestrator_direct
    from report.formatter import ReportFormatter

    print("\n" + "█" * 70)
    print("  ⚕  ClinicalEdge — Pharma Competitive Intelligence Agent")
    print("  Google ADK + MCP Multi-Agent System")
    print("█" * 70)
    print(f"\n  Query: {query}\n")

    result = await run_orchestrator_direct(query)

    if "error" in result:
        print(f"\n❌ Pipeline blocked by guardrails: {result['error']}")
        return

    formatter: ReportFormatter = result["formatter"]

    # Print to terminal
    print()
    formatter.print_rich()

    # Timing summary
    timing = result.get("timing", {})
    print(f"\n  ⏱  Total time:  {timing.get('total_seconds', '?')}s")
    print(f"  ⏱  Agent time:  {timing.get('agent_seconds', '?')}s")

    # Save to file if requested
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext == ".json":
            content = formatter.to_json()
        elif ext in (".md", ".markdown"):
            content = formatter.to_markdown()
        else:
            content = formatter.to_plain_text()

        Path(output_path).write_text(content, encoding="utf-8")
        print(f"\n  📄 Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClinicalEdge — Pharma Competitive Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=DEFAULT_QUERY,
        help="Natural-language competitive intelligence query",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save report to file (supports .md, .json, .txt)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("clinicaledge").setLevel(logging.DEBUG)

    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "\n⚠️  Warning: GOOGLE_API_KEY not set. "
            "Gemini synthesis will use fallback mode.\n"
            "Set the key with:  export GOOGLE_API_KEY='your-api-key'\n"
        )

    asyncio.run(run(args.query, args.output))


if __name__ == "__main__":
    main()

"""CLI: thin wrapper around ``research_trail.cli.run_agent``."""

from __future__ import annotations

import sys

from research_trail.cli import run_agent

if __name__ == "__main__":
    sys.exit(run_agent())

"""CLI: thin wrapper around ``research_trail.cli.run_eval``."""

from __future__ import annotations

import sys

from research_trail.cli import run_eval

if __name__ == "__main__":
    sys.exit(run_eval())

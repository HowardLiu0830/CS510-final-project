"""CLI: thin wrapper around ``research_trail.cli.runs_to_jsonl``."""

from __future__ import annotations

import sys

from research_trail.cli import runs_to_jsonl

if __name__ == "__main__":
    sys.exit(runs_to_jsonl())

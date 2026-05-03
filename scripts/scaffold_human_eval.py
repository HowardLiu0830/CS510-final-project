"""CLI: thin wrapper around ``research_trail.cli.scaffold_human_eval``."""

from __future__ import annotations

import sys

from research_trail.cli import scaffold_human_eval

if __name__ == "__main__":
    sys.exit(scaffold_human_eval())

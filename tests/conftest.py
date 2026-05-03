"""Test fixtures: force offline mode so the harness needs no network."""

import os

os.environ["OPENAI_API_KEY"] = ""

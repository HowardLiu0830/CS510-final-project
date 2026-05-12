"""Test fixtures: force offline mode so the harness needs no network."""

import os

# Both keys must be cleared — offline=True iff both are unset (see
# Settings.offline in src/research_trail/config.py). Otherwise pydantic-settings
# reads OPENROUTER_API_KEY from .env and the harness tries to make real calls.
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

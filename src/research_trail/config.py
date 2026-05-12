"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-5-mini")
    # Set to e.g. "https://openrouter.ai/api/v1" to route OpenAI-compatible
    # traffic through OpenRouter (or any other OpenAI-protocol gateway).
    openai_base_url: str | None = Field(default=None)
    # Optional OpenRouter key. When ``openai_base_url`` points to OpenRouter
    # and this is set, the LLM client uses it in preference to openai_api_key
    # so you don't have to shuffle keys to swap providers.
    openrouter_api_key: str | None = Field(default=None)
    s2_api_key: str | None = Field(default=None)
    openalex_email: str | None = Field(default=None)
    research_trail_cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    extract_concurrency: int = Field(default=8, ge=1, le=32)
    # Number of sub-problems scope_query asks the LLM to produce. The total
    # number of search queries is (1 + n_sub_problems); search_papers caps
    # at this same value after dedup with the main query.
    n_sub_problems: int = Field(default=3, ge=1, le=10)
    # If set, search_papers retrieves only papers published in or before
    # this year. Used for historical-cutoff case studies (e.g., letting
    # an LLM with 2023 knowledge see only 2023 papers, then comparing the
    # predicted gaps against 2024+ literature).
    retrieval_year_max: int | None = Field(default=None, ge=1900, le=2100)

    # When set (e.g. 0.80), build_concept_graph runs embedding-based
    # agglomerative clustering before merging claim/method nodes — semantically
    # equivalent phrases like "GNN" and "graph neural network" collapse into
    # one node instead of staying separate as the pure string-normalize path
    # does. None → keep legacy behaviour (no embedding API calls).
    concept_merge_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # Embedding model to use when concept_merge_threshold is set.
    embedding_model: str = Field(default="text-embedding-3-large")
    # Override where per-run dirs are written. Used by the parallel variant
    # runner so each variant writes to its own data/runs_<variant>/ without
    # polluting baselines.
    runs_root_override: Path | None = Field(default=None)

    @property
    def offline(self) -> bool:
        """True when no live LLM key is configured — agents fall back to stub data.

        Online if *either* key is set, since per-model routing in
        ``llm/client.py`` decides which key to use per call. A run that
        picks a model whose key is missing will fail at request time,
        which is the right failure mode (user error, not silent stub).
        """
        return not (self.openai_api_key or self.openrouter_api_key)


def get_settings() -> Settings:
    s = Settings()
    s.research_trail_cache_dir.mkdir(parents=True, exist_ok=True)
    return s

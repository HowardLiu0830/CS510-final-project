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
    openai_model: str = Field(default="gpt-5-nano")
    s2_api_key: str | None = Field(default=None)
    openalex_email: str | None = Field(default=None)
    research_trail_cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    extract_concurrency: int = Field(default=8, ge=1, le=32)

    @property
    def offline(self) -> bool:
        """True when no live LLM key is configured — agents fall back to stub data."""
        return not self.openai_api_key


def get_settings() -> Settings:
    s = Settings()
    s.research_trail_cache_dir.mkdir(parents=True, exist_ok=True)
    return s

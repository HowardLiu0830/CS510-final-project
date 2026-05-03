"""Base interfaces for academic search clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str


class BaseSearchClient(ABC):
    """Common interface for all academic search clients."""

    source_name: str = "base"

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> list[Paper]:
        """Return up to ``limit`` papers matching ``query``."""
        raise NotImplementedError

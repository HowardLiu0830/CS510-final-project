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
    def search(
        self, query: str, limit: int = 20, *, year_max: int | None = None
    ) -> list[Paper]:
        """Return up to ``limit`` papers matching ``query``.

        ``year_max``: if set, drop any paper whose publication year is later
        than this. Used for retrieval-cutoff ablations. Papers with unknown
        year are dropped conservatively (could be a recent preprint with
        broken metadata).
        """
        raise NotImplementedError


def _passes_year_filter(year: int | None, year_max: int | None) -> bool:
    """Shared post-filter so the three clients enforce the cutoff identically.

    Conservative: drops papers with unknown year when a cutoff is active.
    Without a cutoff, unknown-year papers pass through unchanged.
    """
    if year_max is None:
        return True
    if year is None:
        return False
    return year <= year_max

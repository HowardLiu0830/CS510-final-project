"""Semantic Scholar client."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from research_trail.config import get_settings
from research_trail.search.base import BaseSearchClient, Paper, _passes_year_filter

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient(BaseSearchClient):
    source_name = "semantic_scholar"

    def __init__(self) -> None:
        self._key = get_settings().s2_api_key

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key} if self._key else {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(
        self, query: str, limit: int = 20, *, year_max: int | None = None
    ) -> list[Paper]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,venue,externalIds,url",
        }
        if year_max is not None:
            # S2's `year` param accepts an inclusive range "MIN-MAX".
            params["year"] = f"1900-{year_max}"

        with httpx.Client(timeout=30.0) as c:
            resp = c.get(f"{_S2_BASE}/paper/search", params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json().get("data", [])

        out: list[Paper] = []
        for r in data:
            year = r.get("year")
            if not _passes_year_filter(year, year_max):
                continue
            ext = r.get("externalIds") or {}
            out.append(
                Paper(
                    id=r.get("paperId") or "",
                    title=r.get("title") or "",
                    abstract=r.get("abstract"),
                    year=year,
                    authors=[a.get("name", "") for a in (r.get("authors") or [])],
                    venue=r.get("venue"),
                    doi=ext.get("DOI"),
                    url=r.get("url"),
                    source=self.source_name,
                )
            )
        return out

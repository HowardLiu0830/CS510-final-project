"""OpenAlex client (via pyalex)."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from research_trail.config import get_settings
from research_trail.search.base import BaseSearchClient, Paper


class OpenAlexClient(BaseSearchClient):
    source_name = "openalex"

    def __init__(self) -> None:
        import pyalex

        settings = get_settings()
        if settings.openalex_email:
            pyalex.config.email = settings.openalex_email
        self._pyalex = pyalex

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, limit: int = 20) -> list[Paper]:
        works = self._pyalex.Works().search(query).get(per_page=limit)
        out: list[Paper] = []
        for w in works:
            out.append(
                Paper(
                    id=w.get("id", ""),
                    title=w.get("title") or "",
                    abstract=_invert_abstract(w.get("abstract_inverted_index")),
                    year=w.get("publication_year"),
                    authors=[
                        (a.get("author") or {}).get("display_name", "")
                        for a in w.get("authorships", [])
                    ],
                    venue=(w.get("host_venue") or {}).get("display_name"),
                    doi=w.get("doi"),
                    url=w.get("id"),
                    source=self.source_name,
                )
            )
        return out


def _invert_abstract(idx: dict | None) -> str | None:
    if not idx:
        return None
    positions: list[tuple[int, str]] = []
    for token, locs in idx.items():
        for pos in locs:
            positions.append((pos, token))
    positions.sort()
    return " ".join(tok for _, tok in positions)

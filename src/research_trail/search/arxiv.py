"""arXiv client (via the ``arxiv`` package)."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from research_trail.search.base import BaseSearchClient, Paper


class ArxivClient(BaseSearchClient):
    source_name = "arxiv"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, limit: int = 20) -> list[Paper]:
        import arxiv

        s = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[Paper] = []
        for r in arxiv.Client().results(s):
            out.append(
                Paper(
                    id=r.entry_id,
                    title=r.title,
                    abstract=r.summary,
                    year=r.published.year if r.published else None,
                    authors=[a.name for a in r.authors],
                    venue="arXiv",
                    doi=r.doi,
                    url=r.entry_id,
                    source=self.source_name,
                )
            )
        return out

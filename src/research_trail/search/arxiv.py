"""arXiv client (via the ``arxiv`` package)."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from research_trail.search.base import BaseSearchClient, Paper, _passes_year_filter


class ArxivClient(BaseSearchClient):
    source_name = "arxiv"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(
        self, query: str, limit: int = 20, *, year_max: int | None = None
    ) -> list[Paper]:
        import arxiv

        # arXiv's Lucene-style query language supports a submittedDate range
        # filter. Combined with the user query via AND so we pre-filter at the
        # API level instead of fetching+dropping. Range start = 1900 to keep
        # things permissive on the lower bound.
        if year_max is not None:
            q = f"({query}) AND submittedDate:[19000101 TO {year_max}1231]"
        else:
            q = query

        s = arxiv.Search(
            query=q,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[Paper] = []
        for r in arxiv.Client().results(s):
            year = r.published.year if r.published else None
            # Belt-and-suspenders: also post-filter, since the date-range query
            # syntax is silently lenient on some queries.
            if not _passes_year_filter(year, year_max):
                continue
            out.append(
                Paper(
                    id=r.entry_id,
                    title=r.title,
                    abstract=r.summary,
                    year=year,
                    authors=[a.name for a in r.authors],
                    venue="arXiv",
                    doi=r.doi,
                    url=r.entry_id,
                    source=self.source_name,
                )
            )
        return out

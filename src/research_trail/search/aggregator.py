"""Fan a query across all academic search clients with dedup."""

from __future__ import annotations

import logging

from research_trail.search.arxiv import ArxivClient
from research_trail.search.base import BaseSearchClient, Paper
from research_trail.search.openalex import OpenAlexClient
from research_trail.search.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


def _default_clients() -> list[BaseSearchClient]:
    return [OpenAlexClient(), SemanticScholarClient(), ArxivClient()]


def _dedup_key(p: Paper) -> str:
    if p.doi:
        return f"doi:{p.doi.lower().strip()}"
    return f"title:{p.title.lower().strip()}"


def search_all(
    query: str,
    per_source_limit: int = 10,
    clients: list[BaseSearchClient] | None = None,
) -> list[Paper]:
    """Query each source; suppress per-client failures so one bad source doesn't sink the run."""
    if clients is None:
        try:
            clients = _default_clients()
        except Exception as exc:
            logger.warning("failed to initialize default clients: %s", exc)
            return []

    seen: set[str] = set()
    out: list[Paper] = []
    for client in clients:
        try:
            results = client.search(query, limit=per_source_limit)
        except Exception as exc:
            logger.warning("search failed for %s: %s", type(client).__name__, exc)
            continue
        for p in results:
            key = _dedup_key(p)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out

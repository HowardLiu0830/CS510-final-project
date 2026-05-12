"""Fan a query across all academic search clients with dedup and disk caching."""

from __future__ import annotations

import logging

from research_trail.search.arxiv import ArxivClient
from research_trail.search.base import BaseSearchClient, Paper
from research_trail.search.openalex import OpenAlexClient
from research_trail.search.semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


def _default_clients() -> list[BaseSearchClient]:
    """Build the live search clients, skipping S2 when no API key is set.

    Without a key S2 returns 429 on every request, costing ~30s of retries
    per query while contributing zero papers. Dropping it when keyless
    cuts wall-time substantially without affecting results.
    """
    from research_trail.config import get_settings

    clients: list[BaseSearchClient] = [OpenAlexClient(), ArxivClient()]
    if get_settings().s2_api_key:
        clients.append(SemanticScholarClient())
    return clients


def _dedup_key(p: Paper) -> str:
    if p.doi:
        return f"doi:{p.doi.lower().strip()}"
    return f"title:{p.title.lower().strip()}"


def search_all(
    query: str,
    per_source_limit: int = 10,
    clients: list[BaseSearchClient] | None = None,
    *,
    year_max: int | None = None,
) -> list[Paper]:
    """Query each source; suppress per-client failures so one bad source doesn't sink the run.

    When using the default real clients, results are cached to disk so repeated
    queries across runs avoid redundant API calls (proposal risk mitigation).
    Custom ``clients`` (e.g., in tests) bypass the cache entirely.

    ``year_max``: if set, retrieve only papers published in or before that
    year. Cache key includes the cutoff so a filtered query doesn't share
    cache entries with an unfiltered one.
    """
    use_cache = clients is None
    if clients is None:
        try:
            clients = _default_clients()
        except Exception as exc:
            logger.warning("failed to initialize default clients: %s", exc)
            return []

    cache_dir = None
    if use_cache:
        from research_trail.config import get_settings

        cache_dir = get_settings().research_trail_cache_dir

    from research_trail.search.cache import load_cached, save_cached

    seen: set[str] = set()
    out: list[Paper] = []
    for client in clients:
        try:
            results: list[Paper] | None = None
            if cache_dir is not None:
                results = load_cached(
                    cache_dir, client.source_name, query, per_source_limit, year_max
                )
            if results is None:
                results = client.search(query, limit=per_source_limit, year_max=year_max)
                if cache_dir is not None:
                    save_cached(
                        cache_dir,
                        client.source_name,
                        query,
                        per_source_limit,
                        results,
                        year_max,
                    )
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
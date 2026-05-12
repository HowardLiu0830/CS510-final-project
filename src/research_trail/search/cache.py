"""Disk-based JSON cache for search results.

Keyed by (source_name, query, limit). Results are stored as Paper model dicts.
Cache misses return None; callers fall through to live search.
Writes fail silently so a read-only or slow filesystem never breaks a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from research_trail.search.base import Paper

logger = logging.getLogger(__name__)


def _cache_path(
    cache_dir: Path, source: str, query: str, limit: int, year_max: int | None = None
) -> Path:
    # ``year_max`` is part of the key so a 2020-cutoff query and an
    # unfiltered query don't share results.
    key = hashlib.sha256(f"{source}:{query}:{limit}:y{year_max}".encode()).hexdigest()
    return cache_dir / source / f"{key}.json"


def load_cached(
    cache_dir: Path,
    source: str,
    query: str,
    limit: int,
    year_max: int | None = None,
) -> list[Paper] | None:
    path = _cache_path(cache_dir, source, query, limit, year_max)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        papers = [Paper.model_validate(p) for p in data]
        logger.debug("cache hit: %s q=%r limit=%d -> %d papers", source, query, limit, len(papers))
        return papers
    except Exception as exc:
        logger.debug("cache load failed (%s): %s", path.name, exc)
        return None


def save_cached(
    cache_dir: Path,
    source: str,
    query: str,
    limit: int,
    papers: list[Paper],
    year_max: int | None = None,
) -> None:
    path = _cache_path(cache_dir, source, query, limit, year_max)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([p.model_dump() for p in papers], ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("cache save: %s q=%r -> %d papers", source, query, len(papers))
    except Exception as exc:
        logger.debug("cache write failed (%s): %s", path.name, exc)
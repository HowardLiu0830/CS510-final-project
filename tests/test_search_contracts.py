"""Contract tests for search clients (no network calls required)."""

from research_trail.search.arxiv import ArxivClient
from research_trail.search.base import BaseSearchClient, Paper
from research_trail.search.openalex import OpenAlexClient
from research_trail.search.semantic_scholar import SemanticScholarClient


def test_paper_minimal():
    p = Paper(id="x", title="t", source="stub")
    assert p.id == "x"
    assert p.source == "stub"


def test_clients_are_subclasses():
    assert issubclass(OpenAlexClient, BaseSearchClient)
    assert issubclass(SemanticScholarClient, BaseSearchClient)
    assert issubclass(ArxivClient, BaseSearchClient)

"""Tests for the search aggregator (no network)."""

from __future__ import annotations

from research_trail.search.aggregator import search_all
from research_trail.search.base import BaseSearchClient, Paper


class _FakeClient(BaseSearchClient):
    def __init__(self, name: str, papers: list[Paper]) -> None:
        self.source_name = name
        self._papers = papers

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        return list(self._papers[:limit])


class _BoomClient(BaseSearchClient):
    source_name = "boom"

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        raise RuntimeError("network exploded")


def _p(pid: str, title: str, source: str, doi: str | None = None) -> Paper:
    return Paper(id=pid, title=title, source=source, doi=doi)


def test_aggregator_dedups_by_doi():
    a = _FakeClient("a", [_p("a1", "Same Paper", "a", doi="10.1/x")])
    b = _FakeClient("b", [_p("b1", "Same Paper Different Title Casing", "b", doi="10.1/X")])
    out = search_all("q", clients=[a, b])
    assert len(out) == 1
    assert out[0].id == "a1"


def test_aggregator_dedups_by_title_when_no_doi():
    a = _FakeClient("a", [_p("a1", "  Graph Neural Networks  ", "a")])
    b = _FakeClient("b", [_p("b1", "graph neural networks", "b")])
    out = search_all("q", clients=[a, b])
    assert len(out) == 1


def test_aggregator_suppresses_failing_client():
    a = _BoomClient()
    b = _FakeClient("b", [_p("b1", "Survived", "b")])
    out = search_all("q", clients=[a, b])
    assert len(out) == 1
    assert out[0].id == "b1"


def test_aggregator_returns_empty_when_all_fail():
    out = search_all("q", clients=[_BoomClient(), _BoomClient()])
    assert out == []

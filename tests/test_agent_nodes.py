from research_trail.agents.nodes import search_papers
from research_trail.search.base import Paper


def test_search_papers_caps_number_of_queries(monkeypatch):
    calls = []

    def fake_search_all(query, per_source_limit=3):
        calls.append(query)
        return [
            Paper(
                id=f"{query}-paper",
                title=f"Paper for {query}",
                source="stub",
            )
        ]

    monkeypatch.setattr(
        "research_trail.search.aggregator.search_all",
        fake_search_all,
    )

    state = {
        "query": "main query",
        "sub_problems": [f"sub {i}" for i in range(10)],
    }

    out = search_papers(state)

    assert len(calls) <= 6
    assert len(out["papers"]) <= 6
from research_trail.search.base import Paper
from research_trail.search.cache import load_cached, save_cached


def test_search_cache_save_and_load(tmp_path):
    papers = [
        Paper(
            id="p1",
            title="Test Paper",
            source="stub",
            doi="10.123/test",
        )
    ]

    save_cached(tmp_path, "stub", "graph rag", 3, papers)
    loaded = load_cached(tmp_path, "stub", "graph rag", 3)

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].id == "p1"
    assert loaded[0].doi == "10.123/test"


def test_search_cache_miss_returns_none(tmp_path):
    out = load_cached(tmp_path, "stub", "missing query", 3)

    assert out is None
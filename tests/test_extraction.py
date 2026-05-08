from research_trail.extraction.extractor import extract_from_paper
from research_trail.search.base import Paper


def test_extract_from_paper_offline_returns_stub(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    paper = Paper(
        id="p1",
        title="Graph Neural Networks for Drug Discovery",
        source="stub",
        abstract="This paper studies graph neural networks.",
    )

    out = extract_from_paper(paper)

    assert out.paper_id == "p1"
    assert out.claims
    assert out.methods
    assert out.evidence
    assert out.confidence == 0.0
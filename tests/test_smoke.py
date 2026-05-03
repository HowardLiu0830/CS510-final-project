"""Smoke tests: imports + graph compilation + end-to-end stub run."""

from research_trail import __version__
from research_trail.agents.graph import compile_graph


def test_version():
    assert __version__


def test_compile_graph():
    g = compile_graph()
    assert g is not None


def test_pipeline_runs_offline():
    g = compile_graph()
    state = g.invoke({"query": "graph neural networks for drug discovery"})
    assert state.get("sub_problems")
    assert state.get("papers")
    assert state.get("extractions")
    assert state.get("graph")
    assert "nodes" in state["graph"]
    assert "edges" in state["graph"]
    assert state.get("summary")  # synthesize now produces an offline stub summary

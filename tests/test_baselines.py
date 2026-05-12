"""Smoke tests for the evaluation baselines in offline mode."""

from __future__ import annotations

from research_trail.baselines import run_no_graph, run_zero_shot


def test_zero_shot_offline_returns_full_schema():
    out = run_zero_shot("graph neural networks for drug discovery")
    # Schema parity with the main pipeline output is what lets run_eval consume it.
    assert set(out) == {
        "query",
        "sub_problems",
        "papers",
        "extractions",
        "graph",
        "summary",
        "gaps",
    }
    assert out["summary"]  # offline stub still produces a non-empty string
    assert out["papers"] == [] and out["graph"] == {}  # zero-shot has no retrieval


def test_no_graph_offline_has_papers_but_no_graph():
    out = run_no_graph("graph neural networks for drug discovery")
    assert out["graph"] == {}, "no_graph baseline must not emit a graph"
    # Retrieval + scope + extract still run, so these should be populated by the stubs.
    assert out["sub_problems"]
    assert out["papers"]
    assert out["extractions"]
    assert out["summary"]
    assert out["gaps"]

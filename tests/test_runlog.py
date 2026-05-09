import json

from research_trail.runlog import make_run_dir, serialize_state, write_state
from research_trail.search.base import Paper
from research_trail.extraction.extractor import Extraction


def test_write_state_creates_state_and_graph_json(tmp_path):
    run_dir = make_run_dir("Graph RAG Test", root=tmp_path)

    state = {
        "sub_problems": ["sub problem"],
        "papers": [Paper(id="p1", title="Paper 1", source="stub")],
        "extractions": [Extraction(paper_id="p1", claims=["claim"], methods=["method"])],
        "graph": {
            "nodes": [{"id": "paper:p1", "kind": "paper"}],
            "edges": [],
        },
        "summary": "summary text",
    }

    serial = serialize_state("Graph RAG Test", state)
    write_state(run_dir, serial)

    assert (run_dir / "state.json").exists()
    assert (run_dir / "graph.json").exists()

    state_data = json.loads((run_dir / "state.json").read_text())
    graph_data = json.loads((run_dir / "graph.json").read_text())

    assert state_data["query"] == "Graph RAG Test"
    assert graph_data["nodes"][0]["id"] == "paper:p1"
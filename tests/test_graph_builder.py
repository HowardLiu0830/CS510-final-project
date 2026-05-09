from research_trail.extraction.extractor import Extraction
from research_trail.graph.builder import build_concept_graph


def test_build_graph_creates_paper_claim_method_nodes():
    exts = [
        Extraction(
            paper_id="P1",
            claims=["Graph RAG improves literature exploration"],
            methods=["Knowledge Graph"],
            evidence=["experiment result"],
        )
    ]

    g = build_concept_graph(exts)

    node_ids = {n["id"] for n in g["nodes"]}
    edge_pairs = {(e["source"], e["target"], e["relation"]) for e in g["edges"]}

    assert "paper:P1" in node_ids
    assert "claim:graph rag improves literature exploration" in node_ids
    assert "method:knowledge graph" in node_ids

    assert (
        "paper:P1",
        "claim:graph rag improves literature exploration",
        "asserts",
    ) in edge_pairs

    assert (
        "paper:P1",
        "method:knowledge graph",
        "uses",
    ) in edge_pairs


def test_graph_merges_same_claim_with_different_spacing_and_case():
    exts = [
        Extraction(paper_id="P1", claims=["Graph RAG improves search"]),
        Extraction(paper_id="P2", claims=["  graph rag improves   search  "]),
    ]

    g = build_concept_graph(exts)

    claim_nodes = [n for n in g["nodes"] if n["kind"] == "claim"]

    assert len(claim_nodes) == 1
    assert claim_nodes[0]["id"] == "claim:graph rag improves search"


def test_build_graph_empty_input():
    g = build_concept_graph([])

    assert g["nodes"] == []
    assert g["edges"] == []
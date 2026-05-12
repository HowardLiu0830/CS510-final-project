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


def test_graph_merges_synonyms_with_embedding_threshold(monkeypatch):
    """When concept_merge_threshold is set, semantically equivalent labels
    (faked via a mocked canonical map) collapse into a single node even
    when their string-normalized forms differ.
    """
    from research_trail.graph import embedding

    # Pretend embeddings clustered "GNN" and "graph neural network" together
    # with canonical "graph neural network". Same for drug-discovery variants.
    def fake_canonical(labels, threshold):
        canon = {}
        for lbl in labels:
            lower = lbl.lower()
            if "gnn" in lower or "graph neural network" in lower:
                canon[lbl] = "graph neural network"
            elif "drug discovery" in lower or "drug design" in lower:
                canon[lbl] = "drug discovery"
            else:
                canon[lbl] = lbl
        return canon

    monkeypatch.setattr(embedding, "build_canonical_map", fake_canonical)
    # Flip the threshold so the code-path inside build_concept_graph
    # passes a non-None threshold into our fake.
    monkeypatch.setenv("CONCEPT_MERGE_THRESHOLD", "0.80")

    exts = [
        Extraction(paper_id="P1", claims=["GNN for drug discovery"], methods=["GNN"]),
        Extraction(
            paper_id="P2",
            claims=["Graph Neural Network for drug design"],
            methods=["graph neural network"],
        ),
    ]
    g = build_concept_graph(exts)

    claim_ids = {n["id"] for n in g["nodes"] if n["kind"] == "claim"}
    method_ids = {n["id"] for n in g["nodes"] if n["kind"] == "method"}
    # Both papers' claims should land on the same canonical claim node.
    assert len(claim_ids) == 1
    # Same for methods.
    assert len(method_ids) == 1
    assert "graph neural network" in next(iter(method_ids))
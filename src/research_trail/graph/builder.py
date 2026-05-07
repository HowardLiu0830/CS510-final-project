"""Build a concept graph from extractions."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import networkx as nx

from research_trail.extraction.extractor import Extraction


def _normalize(text: str) -> str:
    """Normalize a concept string for use as a graph node ID.

    Lowercases and collapses whitespace so that "Graph Attention Networks" and
    "graph attention networks" merge into the same node, enabling cross-paper
    linking when two papers use the same concept phrased slightly differently.
    """
    return re.sub(r"\s+", " ", text.lower().strip())


def build_concept_graph(extractions: list[Extraction]) -> dict[str, Any]:
    """Convert extractions into a serialized concept graph (nodes + edges).

    Each paper becomes a node; each unique claim/method becomes a concept node;
    edges connect papers to the concepts they assert/use. Concept node IDs are
    normalized (lowercase + collapsed whitespace) so the same concept mentioned
    by multiple papers merges into one node, revealing shared themes across the
    literature. The display label keeps the first-seen original casing.

    Returns a JSON-able dict suitable for ``streamlit-agraph`` or ``pyvis``.
    """
    g = nx.DiGraph()

    for ext in extractions:
        paper_node = f"paper:{ext.paper_id}"
        g.add_node(paper_node, kind="paper", label=ext.paper_id)
        for claim in ext.claims:
            cn = f"claim:{_normalize(claim)}"
            if not g.has_node(cn):
                g.add_node(cn, kind="claim", label=claim)
            g.add_edge(paper_node, cn, relation="asserts")
        for method in ext.methods:
            mn = f"method:{_normalize(method)}"
            if not g.has_node(mn):
                g.add_node(mn, kind="method", label=method)
            g.add_edge(paper_node, mn, relation="uses")

    return {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"source": u, "target": v, **g.edges[u, v]} for u, v in g.edges],
    }


def build_paper_similarity_graph(extractions: list[Extraction]) -> dict[str, Any]:
    """Build an undirected paper-paper similarity graph.

    Two papers are connected when they share one or more normalized concepts
    (claims or methods). Edge weight = number of shared concepts, which the UI
    maps to line thickness so the strongest relationships stand out.
    """
    # concept → set of paper IDs that mention it
    concept_papers: dict[str, set[str]] = defaultdict(set)
    for ext in extractions:
        for claim in ext.claims:
            concept_papers[_normalize(claim)].add(ext.paper_id)
        for method in ext.methods:
            concept_papers[_normalize(method)].add(ext.paper_id)

    # count shared concepts between each pair
    pair_weight: dict[tuple[str, str], int] = defaultdict(int)
    for papers in concept_papers.values():
        sorted_papers = sorted(papers)
        for i in range(len(sorted_papers)):
            for j in range(i + 1, len(sorted_papers)):
                pair_weight[(sorted_papers[i], sorted_papers[j])] += 1

    g = nx.Graph()
    for ext in extractions:
        g.add_node(f"paper:{ext.paper_id}", kind="paper", label=ext.paper_id)
    for (p1, p2), weight in pair_weight.items():
        g.add_edge(f"paper:{p1}", f"paper:{p2}", weight=weight)

    return {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [
            {"source": u, "target": v, "weight": g.edges[u, v]["weight"]}
            for u, v in g.edges
        ],
    }


def enrich_graph_with_gaps(graph: dict[str, Any], gaps: list[str]) -> dict[str, Any]:
    """Append gap nodes to an already-serialized concept graph.

    Gap nodes use kind="gap" so the UI can colour them distinctly. They are
    added as standalone nodes (no edges) because connecting them to specific
    concepts would require semantic matching that is out of scope here.
    """
    if not gaps:
        return graph
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    existing_ids = {n["id"] for n in nodes}
    for gap in gaps:
        gid = f"gap:{_normalize(gap)}"
        if gid not in existing_ids:
            nodes.append({"id": gid, "kind": "gap", "label": gap})
            existing_ids.add(gid)
    return {"nodes": nodes, "edges": edges}
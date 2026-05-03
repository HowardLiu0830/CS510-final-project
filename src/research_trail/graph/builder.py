"""Build a concept graph from extractions."""

from __future__ import annotations

from typing import Any

import networkx as nx

from research_trail.extraction.extractor import Extraction


def build_concept_graph(extractions: list[Extraction]) -> dict[str, Any]:
    """Convert extractions into a serialized concept graph (nodes + edges).

    Each paper becomes a node; each unique claim/method becomes a concept node;
    edges connect papers to the concepts they assert/use. Returns a JSON-able
    dict suitable for ``streamlit-agraph`` or ``pyvis`` rendering.
    """
    g = nx.DiGraph()

    for ext in extractions:
        paper_node = f"paper:{ext.paper_id}"
        g.add_node(paper_node, kind="paper", label=ext.paper_id)
        for claim in ext.claims:
            cn = f"claim:{claim}"
            g.add_node(cn, kind="claim", label=claim)
            g.add_edge(paper_node, cn, relation="asserts")
        for method in ext.methods:
            mn = f"method:{method}"
            g.add_node(mn, kind="method", label=method)
            g.add_edge(paper_node, mn, relation="uses")

    return {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"source": u, "target": v, **g.edges[u, v]} for u, v in g.edges],
    }

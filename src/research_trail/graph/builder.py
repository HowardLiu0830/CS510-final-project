"""Build a concept graph from extractions."""

from __future__ import annotations

import re
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
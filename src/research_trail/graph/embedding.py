"""Embedding + clustering helpers for the concept graph.

When ``settings.concept_merge_threshold`` is set, ``build_concept_graph``
uses this module to merge semantically-equivalent labels ("GNN" /
"graph neural network", "drug discovery" / "drug design") into the same
node, instead of relying on exact-string normalization alone.

A single batched OpenAI ``embeddings.create`` call covers every label
in the run, then ``sklearn.cluster.AgglomerativeClustering`` groups
labels by cosine similarity ≥ threshold.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from research_trail.config import get_settings

logger = logging.getLogger(__name__)


def embed_concepts(labels: list[str]) -> np.ndarray:
    """Batch-embed a list of label strings, returning an (N, D) array.

    Empty input returns a (0, 0) array so callers can short-circuit
    without a special-case. On API failure we log and re-raise — callers
    catch and fall back to the string-only path.
    """
    if not labels:
        return np.zeros((0, 0), dtype=np.float32)

    settings = get_settings()
    # Route key/base_url the same way the chat client does. We keep this
    # logic local rather than reusing get_chat_model() so we get the raw
    # OpenAI client (LangChain's ChatOpenAI doesn't expose embeddings).
    if settings.openai_base_url and "openrouter" in settings.openai_base_url and settings.openrouter_api_key:
        api_key = settings.openrouter_api_key
        base_url = settings.openai_base_url
    elif settings.openrouter_api_key and "/" in settings.embedding_model:
        # text-embedding-3-* has no slash so we'd take the else branch by
        # default; this branch is only reachable for namespaced embedders.
        api_key = settings.openrouter_api_key
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url  # may be None

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=labels)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vectors


def cluster_labels(
    labels: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> dict[str, str]:
    """Group labels by cosine similarity ≥ threshold.

    Returns a map ``original_label → canonical_label`` so callers can
    rewrite node IDs to the canonical form. Canonical = most frequent
    label within the cluster; ties broken by alphabetical order
    (deterministic so reruns produce identical graphs).
    """
    n = len(labels)
    if n == 0:
        return {}
    if n == 1:
        return {labels[0]: labels[0]}

    from sklearn.cluster import AgglomerativeClustering

    # AgglomerativeClustering's cosine metric expects raw vectors; it
    # computes 1-cosine_similarity internally as the distance. Our
    # threshold is a similarity, so distance_threshold = 1 - threshold.
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    cluster_ids = clusterer.fit_predict(embeddings)

    # Group by cluster_id, pick canonical per cluster.
    groups: dict[int, list[str]] = {}
    for label, cid in zip(labels, cluster_ids):
        groups.setdefault(int(cid), []).append(label)

    canonical_for: dict[str, str] = {}
    for members in groups.values():
        # Most frequent label wins; ties → alphabetical (sorted() is stable).
        counts = Counter(members)
        top_count = max(counts.values())
        candidates = sorted(lbl for lbl, c in counts.items() if c == top_count)
        canonical = candidates[0]
        for member in members:
            canonical_for[member] = canonical
    return canonical_for


def build_canonical_map(labels: list[str], threshold: float | None) -> dict[str, str]:
    """End-to-end: embed labels then cluster them. Returns label → canonical.

    When ``threshold is None`` or labels are empty/singleton, returns
    identity map so callers can use it unconditionally. Logs and falls
    back to identity on any embedding/clustering failure.
    """
    if threshold is None or len(labels) <= 1:
        return {label: label for label in labels}
    unique = sorted(set(labels))  # dedup before embedding (saves API tokens)
    try:
        vectors = embed_concepts(unique)
        return cluster_labels(unique, vectors, threshold)
    except Exception as exc:
        logger.warning("embedding/clustering failed (%s); falling back to identity", exc)
        return {label: label for label in labels}

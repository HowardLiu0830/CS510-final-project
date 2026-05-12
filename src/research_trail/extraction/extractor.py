"""LLM-grounded extraction of claims, methods, and evidence from papers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_trail.config import get_settings
from research_trail.llm.client import get_chat_model
from research_trail.search.base import Paper


class Extraction(BaseModel):
    paper_id: str
    claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    # Short 2-5 word noun phrases naming each claim/method, paralleling the
    # long-text lists by index. Used by build_concept_graph for embedding-
    # based clustering: full-sentence claims rarely match across papers, but
    # their boiled-down keys ("GNN for drug discovery") do. Empty when the
    # extractor didn't produce them (offline stub or legacy runs).
    claim_keys: list[str] = Field(default_factory=list)
    method_keys: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class _RawExtraction(BaseModel):
    claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    claim_keys: list[str] = Field(default_factory=list)
    method_keys: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_PROMPT = """You are a careful research assistant. Read the paper below and extract:
1. Up to 5 key claims (assertions the paper makes), as full sentences.
2. For each claim, a "claim_key" — the **canonical research topic** the claim
   is about, NOT a summary of the claim itself. Use the SHORTEST common name
   a domain expert would use. Strip paper-specific adjectives ("novel",
   "improved", "molecular") unless the modifier defines the topic. Lowercase.
   - GOOD: "graph neural networks", "rlhf alignment", "scaffold-split eval"
   - BAD (too specific): "molecular generative graph neural networks for drug-like compounds"
   - BAD (a summary of the claim): "gnns outperform traditional methods"
   - The same key must be used across papers when they discuss the same topic
     (this is the whole point — we cluster papers by shared topics).
3. Up to 3 methods or techniques used, as the paper names them.
4. For each method, a "method_key" using the same canonical-naming rule.
   - GOOD: "graph convolutional network", "lora", "rotary embedding"
   - BAD: "molecular graph convolutional network for property prediction"
5. Up to 3 pieces of evidence supporting the main claim.
6. Your confidence in the extraction (0.0–1.0): use 0.9+ when the abstract is
   detailed and claims are explicit; use 0.5 or below when it is vague or short.

Return JSON with keys: claims, claim_keys, methods, method_keys, evidence
(all lists of strings; claim_keys parallels claims by index, method_keys
parallels methods) and confidence (float 0.0–1.0).

Title: {title}
Abstract: {abstract}
"""


def extract_from_paper(paper: Paper) -> Extraction:
    """Extract structured knowledge from a single paper.

    On any LLM error we return an empty ``Extraction`` for the paper rather
    than raising — one bad paper shouldn't sink the whole batch when we run
    the extraction concurrently.
    """
    if get_settings().offline:
        return Extraction(
            paper_id=paper.id,
            claims=[f"stub claim about {paper.title[:40]}"],
            claim_keys=["stub claim"],
            methods=["stub method"],
            method_keys=["stub method"],
            evidence=["stub evidence"],
            confidence=0.0,
        )

    model = get_chat_model()
    if model is None:
        return Extraction(paper_id=paper.id)
    try:
        structured = model.with_structured_output(_RawExtraction)
        raw: _RawExtraction = structured.invoke(
            _PROMPT.format(title=paper.title, abstract=paper.abstract or "")
        )
        # Pad keys to match claims/methods length so callers can index by
        # position. If the LLM under-produces keys, fall back to using the
        # full claim/method text as the key for that index.
        claim_keys = list(raw.claim_keys)
        while len(claim_keys) < len(raw.claims):
            claim_keys.append(raw.claims[len(claim_keys)])
        method_keys = list(raw.method_keys)
        while len(method_keys) < len(raw.methods):
            method_keys.append(raw.methods[len(method_keys)])
        return Extraction(
            paper_id=paper.id,
            claims=raw.claims,
            claim_keys=claim_keys[: len(raw.claims)],
            methods=raw.methods,
            method_keys=method_keys[: len(raw.methods)],
            evidence=raw.evidence,
            confidence=round(raw.confidence, 3),
        )
    except Exception:
        return Extraction(paper_id=paper.id, confidence=0.0)

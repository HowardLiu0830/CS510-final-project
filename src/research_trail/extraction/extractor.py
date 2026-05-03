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
    confidence: float = 1.0


class _RawExtraction(BaseModel):
    claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


_PROMPT = """You are a careful research assistant. Read the paper below and extract:
1. Up to 5 key claims (assertions the paper makes)
2. Up to 3 methods or techniques used
3. Up to 3 pieces of evidence supporting the main claim

Return JSON with keys: claims, methods, evidence (each a list of strings).

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
            methods=["stub method"],
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
        return Extraction(
            paper_id=paper.id,
            claims=raw.claims,
            methods=raw.methods,
            evidence=raw.evidence,
        )
    except Exception:
        return Extraction(paper_id=paper.id, confidence=0.0)

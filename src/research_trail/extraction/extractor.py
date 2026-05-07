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
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


_PROMPT = """You are a careful research assistant. Read the paper below and extract:
1. Up to 5 key claims (assertions the paper makes)
2. Up to 3 methods or techniques used
3. Up to 3 pieces of evidence supporting the main claim
4. Your confidence in the extraction (0.0–1.0): use 0.9+ when the abstract is
   detailed and claims are explicit; use 0.5 or below when it is vague or very short.

Return JSON with keys: claims, methods, evidence (lists of strings) and
confidence (a float 0.0–1.0).

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
            confidence=round(raw.confidence, 3),
        )
    except Exception:
        return Extraction(paper_id=paper.id, confidence=0.0)

"""Rubric definition.

Four dimensions, each on a 1-5 scale, mirroring the proposal:
- relevance: results match the user's query intent
- coverage: breadth of subtopics covered
- structural_organization: graph layout reveals real conceptual structure
- insightfulness: surfacing of non-obvious connections / gaps
"""

from __future__ import annotations

from pydantic import BaseModel, Field

DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "coverage",
    "structural_organization",
    "insightfulness",
)


class RubricScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    coverage: int = Field(ge=1, le=5)
    structural_organization: int = Field(ge=1, le=5)
    insightfulness: int = Field(ge=1, le=5)
    rationale: str = ""

    @property
    def mean(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)


class Rubric(BaseModel):
    """A single judge's evaluation of a system run."""

    judge_id: str
    query: str
    score: RubricScore

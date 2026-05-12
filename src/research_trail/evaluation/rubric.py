"""Rubric definition.

The system produces two user-facing artefacts — a literature review
(``summary``) and a list of research gaps (``gaps``). Each artefact has
its own 4-dimension rubric; the concept graph is an intermediate
representation, judged indirectly via the downstream artefact quality.

Eight dimensions total, each on a 1-5 scale with anchored 1/3/5
descriptions (see ``LEVEL_ANCHORS``). The judge prompt embeds these
anchors verbatim so the LLM grades against the same definitions a
human reviewer sees.

Literature review (judges ``summary``):
- review_relevance     : content matches the user's query intent
- review_coverage      : breadth of subtopics covered
- review_groundedness  : claims supported by retrieved papers (no hallucination)
- review_synthesis     : findings woven together vs. enumerated paper-by-paper

Research gaps (judges ``gaps``):
- gap_relevance     : gaps fall within the query's research area
- gap_specificity   : each gap is a concrete, actionable research direction
- gap_novelty       : gaps are non-obvious, not well-trodden open problems
- gap_significance  : filling the gap would meaningfully advance the field
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Grouped for downstream consumers (UI sections, per-artefact averages).
REVIEW_DIMENSIONS: tuple[str, ...] = (
    "review_relevance",
    "review_coverage",
    "review_groundedness",
    "review_synthesis",
)
GAP_DIMENSIONS: tuple[str, ...] = (
    "gap_relevance",
    "gap_specificity",
    "gap_novelty",
    "gap_significance",
)
DIMENSIONS: tuple[str, ...] = REVIEW_DIMENSIONS + GAP_DIMENSIONS


# Anchored level descriptions per dimension. Each dim has all five levels
# defined so the judge does not have to interpolate. The judge prompt is
# built directly from this table so prompt and schema stay in sync.
LEVEL_ANCHORS: dict[str, dict[int, str]] = {
    # ── Literature review ────────────────────────────────────────────
    "review_relevance": {
        1: "Off-topic; barely matches the query intent.",
        2: "Mostly off-topic, with a few on-topic paragraphs.",
        3: "Roughly half on-topic; the rest drifts to tangential areas.",
        4: "Mostly on-topic with one or two tangential digressions.",
        5: "Entire review directly addresses the query and its sub-aspects.",
    },
    "review_coverage": {
        1: "Only one narrow angle; major obvious subtopics are missing.",
        2: "Two or three subtopics covered; most expected breadth is missing.",
        3: "Main subtopics covered but misses one or two important angles.",
        4: "Nearly all expected subtopics covered; one minor angle missing.",
        5: "Comprehensively covers the subtopics a domain expert would expect.",
    },
    "review_groundedness": {
        1: "Most claims appear fabricated; cited papers / titles not in the retrieved set.",
        2: "Many claims unsupported; only some titles appear in the retrieved set.",
        3: "Some claims grounded in retrieved papers; others speculative or uncited.",
        4: "Most claims grounded; one or two unsupported assertions remain.",
        5: "Every substantive claim is traceable to a retrieved paper; no apparent fabrication.",
    },
    "review_synthesis": {
        1: "Reads like an annotated bibliography — each paper described in isolation.",
        2: "Mostly paper-by-paper, with sporadic transitions between them.",
        3: "Some integration of findings, but mostly paper-by-paper enumeration.",
        4: "Findings mostly integrated across papers; a few isolated descriptions remain.",
        5: "Findings woven across papers — comparisons, contrasts, and shared themes are explicit.",
    },
    # ── Research gaps ────────────────────────────────────────────────
    "gap_relevance": {
        1: "Gaps are about a different research area than the query.",
        2: "Most gaps tangential; one or two are query-relevant.",
        3: "Some gaps query-relevant; others tangential.",
        4: "Most gaps in-domain; one drifts.",
        5: "Every gap squarely within the query's research area.",
    },
    "gap_specificity": {
        1: "Vague platitudes ('more research needed', 'better understanding required').",
        2: "Mostly vague; one or two name a specific area but no method.",
        3: "Names a topic or method but stops short of a concrete, testable direction.",
        4: "Most gaps concrete; one or two remain abstract.",
        5: "Each gap names a specific research question, method, dataset, or comparison a researcher could act on.",
    },
    "gap_novelty": {
        1: "Repeats well-known open problems verbatim, or claims something open that is already solved.",
        2: "Mostly textbook open problems with little fresh framing.",
        3: "Real open questions but obvious to anyone familiar with the area.",
        4: "Most gaps non-trivial; one or two are well-known.",
        5: "Surfaces non-obvious open questions visible only when reading the retrieved literature together.",
    },
    "gap_significance": {
        1: "Even if filled, these gaps would have negligible impact on the field.",
        2: "Filling would produce minor or narrow contributions.",
        3: "Would yield incremental contributions if addressed.",
        4: "Most gaps meaningful; addressing them would notably advance some area.",
        5: "Filling these gaps would meaningfully advance the field or open new directions.",
    },
}


class RubricScore(BaseModel):
    # Literature review dims.
    review_relevance: int = Field(ge=1, le=5)
    review_coverage: int = Field(ge=1, le=5)
    review_groundedness: int = Field(ge=1, le=5)
    review_synthesis: int = Field(ge=1, le=5)

    # Research gap dims.
    gap_relevance: int = Field(ge=1, le=5)
    gap_specificity: int = Field(ge=1, le=5)
    gap_novelty: int = Field(ge=1, le=5)
    gap_significance: int = Field(ge=1, le=5)

    # One-sentence justification per dimension — written *before* the
    # integer score so the judge "thinks out loud", which empirically
    # improves consistency.
    review_relevance_note: str = ""
    review_coverage_note: str = ""
    review_groundedness_note: str = ""
    review_synthesis_note: str = ""
    gap_relevance_note: str = ""
    gap_specificity_note: str = ""
    gap_novelty_note: str = ""
    gap_significance_note: str = ""

    # Optional free-text overall rationale; kept for backward-compat
    # with the old single-rationale human-eval forms.
    rationale: str = ""

    @property
    def mean(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)

    @property
    def review_mean(self) -> float:
        return sum(getattr(self, d) for d in REVIEW_DIMENSIONS) / len(REVIEW_DIMENSIONS)

    @property
    def gap_mean(self) -> float:
        return sum(getattr(self, d) for d in GAP_DIMENSIONS) / len(GAP_DIMENSIONS)


class Rubric(BaseModel):
    """A single judge's evaluation of a system run."""

    judge_id: str
    query: str
    score: RubricScore

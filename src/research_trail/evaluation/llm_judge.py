"""Rubric-guided LLM-as-judge evaluator.

Two artefacts are graded with separate 4-dimension rubrics: the
literature review (``summary``) and the research gaps (``gaps``). The
judge prompt embeds anchored 1/3/5 level descriptions per dimension
(see ``rubric.LEVEL_ANCHORS``) and requires the model to write a
one-sentence justification per dimension *before* committing to an
integer score. This "anchor + reason-then-score" pattern is the
standard recipe for rubric-guided LLM eval (G-Eval, MT-Bench) and
substantially reduces the model's tendency to default to mid-scale
scores.
"""

from __future__ import annotations

import json

from research_trail.config import get_settings
from research_trail.evaluation.rubric import (
    GAP_DIMENSIONS,
    LEVEL_ANCHORS,
    REVIEW_DIMENSIONS,
    Rubric,
    RubricScore,
)
from research_trail.llm.client import get_chat_model


def _render_section(title: str, dims: tuple[str, ...]) -> str:
    lines = [f"## {title}", ""]
    for dim in dims:
        anchors = LEVEL_ANCHORS[dim]
        lines.append(f"### {dim}")
        for level in sorted(anchors):
            lines.append(f"  {level} — {anchors[level]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_rubric_block() -> str:
    """Render the two anchored sub-rubrics as plain text for the prompt."""
    return (
        _render_section("Literature review rubric (judges `summary`)", REVIEW_DIMENSIONS)
        + "\n\n"
        + _render_section("Research gaps rubric (judges `gaps`)", GAP_DIMENSIONS)
    )


_JUDGE_PROMPT = """You are an expert research librarian evaluating a literature-
exploration system. The system produces two artefacts; grade them with the
two separate rubrics below.

- ``summary``: a written literature review. Graded on review_* dimensions.
- ``gaps``: a list of identified research gaps. Graded on gap_* dimensions.

The retrieved ``papers`` are the source of truth for groundedness checks.
The ``graph`` field, if present, is an intermediate representation — do not
grade it directly, but let it inform your reading of the two artefacts.

Score each dimension on a 1-5 integer scale. All five levels are anchored
below — pick the level whose description best matches your evidence.

{rubric_block}

## How to grade

1. For each dimension, first write ONE concise sentence (the ``*_note`` field)
   that cites concrete evidence from the system output — e.g. "names 4 of the
   5 major subtopics", "review enumerates papers without comparing findings",
   "Gap #2 says 'more interpretability work needed' without a method".
2. THEN choose the integer score that best matches your note against the
   anchored levels. Do not bias toward the middle; use 1 or 5 when warranted.
3. Finally write a short overall ``rationale`` (1-2 sentences) summarising the
   biggest strength and weakness across both artefacts.

If ``gaps`` is empty or missing, grade every gap_* dimension as 1 — the
absence of gaps IS the measurement.

## Query

{query}

## System output (JSON, possibly truncated)

{output}
"""


def judge(query: str, system_output: dict) -> Rubric:
    settings = get_settings()
    judge_id = f"llm:{settings.openai_model}"

    if settings.offline:
        # Deterministic offline stub so tests + offline smoke-runs work.
        stub_fields: dict = {dim: 3 for dim in REVIEW_DIMENSIONS + GAP_DIMENSIONS}
        stub_fields.update({f"{dim}_note": "offline stub" for dim in stub_fields})
        stub_fields["rationale"] = "offline stub — no real evaluation performed"
        return Rubric(
            judge_id=f"{judge_id}:offline-stub",
            query=query,
            score=RubricScore(**stub_fields),
        )

    model = get_chat_model()
    structured = model.with_structured_output(RubricScore)
    prompt = _JUDGE_PROMPT.format(
        rubric_block=_render_rubric_block(),
        query=query,
        output=_compact_payload(system_output),
    )
    score: RubricScore = structured.invoke(prompt)
    return Rubric(judge_id=judge_id, query=query, score=score)


# Per-field char budgets so the judge always sees what it has to grade.
# A naive ``json.dumps(state)[:6000]`` truncates after ``graph`` (the
# biggest field) and silently drops ``summary``/``gaps`` from the tail.
_PAYLOAD_BUDGETS = {
    "summary": 5000,    # the literature review itself, graded in full
    "gaps": 2000,       # 3-5 short strings
    "paper_titles": 3000,  # title list = source of truth for groundedness
}


def _compact_payload(state: dict) -> str:
    """Build a judge-targeted JSON payload from a full pipeline state.

    Includes only the fields the rubric actually grades, with per-field
    budgets so a long graph doesn't crowd out the review and gaps.
    """
    summary = (state.get("summary") or "")[: _PAYLOAD_BUDGETS["summary"]]
    gaps = state.get("gaps") or []
    # Titles only — judge uses them for review_groundedness / gap_relevance
    # checks; full paper objects would blow the budget.
    titles = [(p.get("title") or "")[:200] for p in (state.get("papers") or [])]
    titles_json = json.dumps(titles, ensure_ascii=False)
    if len(titles_json) > _PAYLOAD_BUDGETS["paper_titles"]:
        # Keep as many full titles as fit, drop the tail rather than
        # truncating mid-title (which would corrupt grounding checks).
        kept: list[str] = []
        running = 2  # for the [] braces
        for t in titles:
            piece = json.dumps(t, ensure_ascii=False) + ","
            if running + len(piece) > _PAYLOAD_BUDGETS["paper_titles"]:
                break
            kept.append(t)
            running += len(piece)
        titles = kept

    payload = {
        "summary": summary,
        "gaps": gaps[: _PAYLOAD_BUDGETS["gaps"] // 200],  # cap count too
        "paper_titles": titles,
        "n_papers_total": len(state.get("papers") or []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

"""LLM-as-judge evaluator."""

from __future__ import annotations

import json

from research_trail.config import get_settings
from research_trail.evaluation.rubric import Rubric, RubricScore
from research_trail.llm.client import get_chat_model

_JUDGE_PROMPT = """You are an expert research librarian evaluating a system's
output for a literature-exploration query. Score each dimension on a 1-5 scale
where 1 is poor and 5 is excellent.

Dimensions:
- relevance: how well do retrieved/extracted items match the query?
- coverage: breadth across subtopics
- structural_organization: does the concept graph reveal real structure?
- insightfulness: are non-obvious links/gaps surfaced?

Query:
{query}

System output (JSON, possibly truncated):
{output}

Return JSON with integer fields relevance, coverage, structural_organization,
insightfulness (each 1-5), and a short string field rationale.
"""


def judge(query: str, system_output: dict) -> Rubric:
    settings = get_settings()
    judge_id = f"llm:{settings.openai_model}"

    if settings.offline:
        return Rubric(
            judge_id=f"{judge_id}:offline-stub",
            query=query,
            score=RubricScore(
                relevance=3,
                coverage=3,
                structural_organization=3,
                insightfulness=3,
                rationale="offline stub — no real evaluation performed",
            ),
        )

    model = get_chat_model()
    structured = model.with_structured_output(RubricScore)
    score: RubricScore = structured.invoke(
        _JUDGE_PROMPT.format(query=query, output=json.dumps(system_output)[:6000])
    )
    return Rubric(judge_id=judge_id, query=query, score=score)

"""No-graph baseline: full retrieval pipeline minus the concept graph.

Runs scope -> search -> screen-and-extract -> synthesize -> identify gaps,
but never builds the graph and never feeds graph statistics into prompts.
Synthesis and gap-finding see only the papers + extractions, so any
difference vs. the full pipeline isolates the graph's contribution.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from research_trail.agents.nodes import (
    scope_query,
    screen_and_extract,
    search_papers,
)
from research_trail.config import get_settings
from research_trail.llm.client import get_chat_model
from research_trail.runlog import node

_SYNTH_PROMPT = """You are a research librarian writing a brief literature
survey for the user's query. Use the structured findings below to produce a
3-5 paragraph synthesis that names the main subtopics, the most relevant
papers, and any apparent gaps. Cite papers inline by their title.

User query:
{query}

Sub-problems:
{sub_problems}

Top papers (JSON):
{papers}

Write the synthesis as plain text, no markdown headers.
"""


_GAP_PROMPT = """You are a research librarian conducting a systematic literature
review. Based on the synthesis below, identify 3-5 concrete research gaps —
areas that are understudied, have contradictory findings, or where current
methods fall short.

Query: {query}

Synthesis:
{summary}

Return JSON with one field: gaps (a list of 3-5 strings).
"""


class _Gaps(BaseModel):
    gaps: list[str] = Field(default_factory=list)


@node("synthesize_no_graph")
def _synthesize(state: dict) -> dict:
    query = state.get("query", "")
    if get_settings().offline:
        n_papers = len(state.get("papers", []))
        n_subs = len(state.get("sub_problems", []))
        return {
            "summary": (
                f"[offline stub] no-graph synthesis for '{query}': "
                f"{n_subs} sub-problems, {n_papers} papers retrieved."
            )
        }

    model = get_chat_model()
    if model is None:
        return {"summary": ""}

    papers = state.get("papers", [])
    paper_payload = [
        {
            "title": p.title,
            "authors": p.authors[:5],
            "year": p.year,
            "abstract": (p.abstract or "")[:500],
        }
        for p in papers[:10]
    ]
    prompt = _SYNTH_PROMPT.format(
        query=query,
        sub_problems="\n".join(f"- {s}" for s in state.get("sub_problems", [])),
        papers=json.dumps(paper_payload)[:6000],
    )
    try:
        resp = model.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        return {"summary": text}
    except Exception as exc:
        return {"summary": f"[synthesis failed: {exc}]"}


@node("identify_gaps_no_graph")
def _identify_gaps(state: dict) -> dict:
    query = state.get("query", "")
    summary = state.get("summary") or ""

    if get_settings().offline:
        return {"gaps": [f"Open question in '{query}': further study needed (stub)"]}

    model = get_chat_model()
    if model is None:
        return {"gaps": []}
    try:
        structured = model.with_structured_output(_Gaps)
        out: _Gaps = structured.invoke(
            _GAP_PROMPT.format(query=query, summary=summary[:3000])
        )
        return {"gaps": [g.strip() for g in out.gaps if g.strip()]}
    except Exception:
        return {"gaps": []}


def run_no_graph(query: str) -> dict:
    """Run the no-graph baseline and return a state-shaped dict.

    The graph field is left empty on purpose — that is the ablation.
    """
    state: dict = {"query": query}
    state.update(scope_query(state))
    state.update(search_papers(state))
    state.update(screen_and_extract(state))
    state.update(_synthesize(state))
    state.update(_identify_gaps(state))
    return {
        "query": query,
        "sub_problems": state.get("sub_problems", []),
        "papers": state.get("papers", []),
        "extractions": state.get("extractions", []),
        "graph": {},
        "summary": state.get("summary", ""),
        "gaps": state.get("gaps", []),
    }

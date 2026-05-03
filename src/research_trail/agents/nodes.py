"""LangGraph nodes.

Each node takes the current state and returns a partial update dict. Live mode
calls real services (OpenAI + academic-search clients); offline mode (no
``OPENAI_API_KEY``) returns deterministic stubs so the graph compiles and runs
end-to-end without network.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from research_trail.config import get_settings
from research_trail.llm.client import get_chat_model
from research_trail.runlog import node
from research_trail.search.base import Paper


class _SubProblems(BaseModel):
    sub_problems: list[str] = Field(default_factory=list)


_SCOPE_PROMPT = """You are a research strategist. Decompose the user's query
into 3-5 concrete sub-problems that, taken together, would produce a
well-rounded literature survey. Each sub-problem should be a short noun
phrase or question that could itself be searched against an academic index.

User query:
{query}

Return JSON with one field: sub_problems (a list of 3-5 strings).
"""


@node("scope_query")
def scope_query(state: dict) -> dict:
    """Decompose the query into sub-problems."""
    query = state.get("query", "")
    if get_settings().offline:
        return {"sub_problems": [f"sub-problem: {query} (stub)"]}

    model = get_chat_model()
    if model is None:
        return {"sub_problems": [query]}
    try:
        structured = model.with_structured_output(_SubProblems)
        out: _SubProblems = structured.invoke(_SCOPE_PROMPT.format(query=query))
        subs = [s.strip() for s in out.sub_problems if s and s.strip()]
        return {"sub_problems": subs or [query]}
    except Exception:
        return {"sub_problems": [query]}


@node("search_papers")
def search_papers(state: dict) -> dict:
    """Search academic databases for relevant papers."""
    query = state.get("query", "")
    if get_settings().offline:
        return {
            "papers": [
                Paper(
                    id="stub-1",
                    title=f"Stub paper on {query}",
                    abstract="Placeholder abstract.",
                    year=2024,
                    authors=["A. Researcher"],
                    source="stub",
                    url="https://example.org/stub-1",
                )
            ]
        }
    from research_trail.search.aggregator import search_all

    return {"papers": search_all(query, per_source_limit=10)}


@node("screen_and_extract")
def screen_and_extract(state: dict) -> dict:
    """Screen retrieved papers and extract claims/methods."""
    from research_trail.extraction.extractor import Extraction, extract_from_paper

    papers: list[Paper] = state.get("papers", [])
    if get_settings().offline:
        return {
            "extractions": [
                Extraction(
                    paper_id=p.id,
                    claims=[f"stub claim for {p.title[:40]}"],
                    methods=["stub method"],
                    evidence=["stub evidence"],
                    confidence=0.0,
                )
                for p in papers
            ]
        }
    return {"extractions": [extract_from_paper(p) for p in papers]}


@node("build_graph")
def build_graph(state: dict) -> dict:
    """Construct the concept graph from extractions."""
    from research_trail.graph.builder import build_concept_graph

    return {"graph": build_concept_graph(state.get("extractions", []))}


_SYNTHESIZE_PROMPT = """You are a research librarian writing a brief literature
survey for the user's query. Use the structured findings below to produce a
3-5 paragraph synthesis that names the main subtopics, the most relevant
papers, and any apparent gaps. Cite papers inline by their title.

User query:
{query}

Sub-problems:
{sub_problems}

Top papers (JSON):
{papers}

Concept-graph stats:
{graph_stats}

Write the synthesis as plain text, no markdown headers.
"""


@node("synthesize")
def synthesize(state: dict) -> dict:
    """Produce a written summary tying sub-problems, papers, and graph together."""
    query = state.get("query", "")
    if get_settings().offline:
        n_papers = len(state.get("papers", []))
        n_subs = len(state.get("sub_problems", []))
        return {
            "summary": (
                f"[offline stub] Synthesis for '{query}': "
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
    graph = state.get("graph", {}) or {}
    graph_stats = {
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
    }
    prompt = _SYNTHESIZE_PROMPT.format(
        query=query,
        sub_problems="\n".join(f"- {s}" for s in state.get("sub_problems", [])),
        papers=json.dumps(paper_payload)[:6000],
        graph_stats=json.dumps(graph_stats),
    )
    try:
        resp = model.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        return {"summary": text}
    except Exception as exc:
        return {"summary": f"[synthesis failed: {exc}]"}

"""Zero-shot baseline: ask the LLM the query directly, no retrieval or graph."""

from __future__ import annotations

from research_trail.config import get_settings
from research_trail.llm.client import get_chat_model
from research_trail.runlog import node

_ZERO_SHOT_PROMPT = """You are a research librarian. The user has asked a
literature-exploration question. Without access to any external retrieval
system, write a 3-5 paragraph literature survey from your own knowledge.
Name subtopics, mention representative papers/authors if you can recall
them, and call out apparent gaps or open questions.

User query:
{query}

Write plain text, no markdown headers.
"""


@node("zero_shot")
def _zero_shot_call(state: dict) -> dict:
    query = state.get("query", "")
    if get_settings().offline:
        return {"summary": f"[offline stub] zero-shot answer for '{query}'."}

    model = get_chat_model()
    if model is None:
        return {"summary": ""}
    try:
        resp = model.invoke(_ZERO_SHOT_PROMPT.format(query=query))
        text = getattr(resp, "content", None) or str(resp)
        return {"summary": text}
    except Exception as exc:
        return {"summary": f"[zero-shot failed: {exc}]"}


def run_zero_shot(query: str) -> dict:
    """Run the zero-shot baseline and return a state-shaped dict.

    Missing pipeline fields (sub_problems, papers, extractions, graph, gaps)
    are filled with their empty defaults so downstream tooling that expects
    the full schema (run_eval, runs_to_jsonl) works without special-casing.
    """
    out = _zero_shot_call({"query": query})
    return {
        "query": query,
        "sub_problems": [],
        "papers": [],
        "extractions": [],
        "graph": {},
        "summary": out.get("summary", ""),
        "gaps": [],
    }

"""Compile the research-trail LangGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from research_trail.agents.nodes import (
    build_graph,
    scope_query,
    screen_and_extract,
    search_papers,
    synthesize,
)
from research_trail.agents.state import ResearchState


def compile_graph():
    sg = StateGraph(ResearchState)
    sg.add_node("scope_query", scope_query)
    sg.add_node("search_papers", search_papers)
    sg.add_node("screen_and_extract", screen_and_extract)
    sg.add_node("build_graph", build_graph)
    sg.add_node("synthesize", synthesize)

    sg.add_edge(START, "scope_query")
    sg.add_edge("scope_query", "search_papers")
    sg.add_edge("search_papers", "screen_and_extract")
    sg.add_edge("screen_and_extract", "build_graph")
    sg.add_edge("build_graph", "synthesize")
    sg.add_edge("synthesize", END)

    return sg.compile()

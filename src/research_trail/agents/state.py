"""Shared state schema for the research-trail LangGraph."""

from __future__ import annotations

from typing import TypedDict

from research_trail.extraction.extractor import Extraction
from research_trail.search.base import Paper


class ResearchState(TypedDict, total=False):
    query: str
    sub_problems: list[str]
    papers: list[Paper]
    extractions: list[Extraction]
    graph: dict
    summary: str
    judgments: dict

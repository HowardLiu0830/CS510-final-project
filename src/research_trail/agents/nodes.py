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
into EXACTLY {n} concrete sub-problem(s) that, taken together, would produce a
well-rounded literature survey. Each sub-problem should be a short noun
phrase or question that could itself be searched against an academic index.

User query:
{query}

Return JSON with one field: sub_problems (a list of exactly {n} string(s)).
"""


@node("scope_query")
def scope_query(state: dict) -> dict:
    """Decompose the query into sub-problems."""
    query = state.get("query", "")
    settings = get_settings()
    n = settings.n_sub_problems
    if settings.offline:
        return {"sub_problems": [f"sub-problem {i+1}: {query} (stub)" for i in range(n)]}

    model = get_chat_model()
    if model is None:
        return {"sub_problems": [query]}
    try:
        structured = model.with_structured_output(_SubProblems)
        out: _SubProblems = structured.invoke(_SCOPE_PROMPT.format(query=query, n=n))
        subs = [s.strip() for s in out.sub_problems if s and s.strip()][:n]
        return {"sub_problems": subs or [query]}
    except Exception:
        return {"sub_problems": [query]}


@node("search_papers")
def search_papers(state: dict) -> dict:
    """Search academic databases for relevant papers.

    Searches the main query and each sub-problem separately, then deduplicates
    across all searches so that diverse sub-topics each get their own coverage.
    Results from each (query, source) pair are disk-cached to avoid redundant
    API calls on repeated or overlapping sub-problems.
    """
    query = state.get("query", "")
    sub_problems = state.get("sub_problems", [])
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
    from concurrent.futures import ThreadPoolExecutor

    from research_trail.search.aggregator import search_all

    # Search the main query and each sub-problem in parallel, then deduplicate.
    # Cap = 1 + n_sub_problems (matches scope_query's contract).
    settings = get_settings()
    cap = 1 + settings.n_sub_problems
    year_max = settings.retrieval_year_max
    queries = list(dict.fromkeys([query] + sub_problems))[:cap]
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results_per_query = list(pool.map(
            lambda q: search_all(q, per_source_limit=3, year_max=year_max),
            queries,
        ))

    seen_keys: set[str] = set()
    all_papers: list[Paper] = []
    for papers in results_per_query:
        for p in papers:
            key = f"doi:{p.doi.lower().strip()}" if p.doi else f"title:{p.title.lower().strip()}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_papers.append(p)
    return {"papers": all_papers}


@node("screen_and_extract")
def screen_and_extract(state: dict) -> dict:
    """Screen retrieved papers and extract claims/methods.

    Live mode runs ``extract_from_paper`` concurrently across papers because
    each call is dominated by an OpenAI round-trip (10-20s) — serial execution
    on 20 papers is the dominant bottleneck of the whole pipeline. Order is
    preserved so ``extractions[i]`` still corresponds to ``papers[i]``.
    """
    from concurrent.futures import ThreadPoolExecutor

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
    if not papers:
        return {"extractions": []}
    workers = min(get_settings().extract_concurrency, len(papers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extractions = list(pool.map(extract_from_paper, papers))
    return {"extractions": extractions}


class _Gaps(BaseModel):
    gaps: list[str] = Field(default_factory=list)


_GAP_PROMPT = """You are a research librarian conducting a systematic literature review.
Based on the synthesis and concept-graph statistics below, identify 3-5 concrete
research gaps — areas that are understudied, have contradictory findings, or where
current methods fall short. Ground each gap in the evidence: reference specific
concepts or methods where possible.

Query: {query}

Synthesis:
{summary}

Concept-graph statistics:
- {n_papers} papers, {n_concepts} distinct concepts, {n_edges} relationships
- Most cross-referenced concepts: {top_concepts}
- Concepts cited by only one paper (potentially niche): {niche_concepts}

Return JSON with one field: gaps (a list of 3-5 strings).
"""


def _graph_stats(graph: dict) -> dict:
    """Compute concept-frequency stats from a serialised graph dict."""
    from collections import Counter

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    concept_counts: Counter[str] = Counter()
    for e in edges:
        if e.get("source", "").startswith("paper:"):
            concept_counts[e.get("target", "")] += 1
    node_label = {n["id"]: n.get("label", n["id"]) for n in nodes}
    top = [node_label[cid] for cid, _ in concept_counts.most_common(8) if cid in node_label]
    niche = [
        node_label[cid]
        for cid, cnt in concept_counts.items()
        if cnt == 1 and cid in node_label
    ][:5]
    return {
        "n_papers": sum(1 for n in nodes if n.get("kind") == "paper"),
        "n_concepts": sum(1 for n in nodes if n.get("kind") in ("claim", "method")),
        "n_edges": len(edges),
        "top_concepts": ", ".join(top) or "(none)",
        "niche_concepts": ", ".join(niche) or "(none)",
    }


@node("identify_gaps")
def identify_gaps(state: dict) -> dict:
    """Surface research gaps from the synthesized literature and concept graph."""
    from research_trail.graph.builder import enrich_graph_with_gaps

    query = state.get("query", "")
    graph = state.get("graph", {}) or {}

    if get_settings().offline:
        gaps = [f"Open question in '{query}': further study needed (stub)"]
        return {"gaps": gaps, "graph": enrich_graph_with_gaps(graph, gaps)}

    model = get_chat_model()
    if model is None:
        return {"gaps": [], "graph": graph}
    try:
        structured = model.with_structured_output(_Gaps)
        out: _Gaps = structured.invoke(
            _GAP_PROMPT.format(
                query=query,
                summary=(state.get("summary") or "")[:3000],
                **_graph_stats(graph),
            )
        )
        gaps = [g.strip() for g in out.gaps if g.strip()]
        return {"gaps": gaps, "graph": enrich_graph_with_gaps(graph, gaps)}
    except Exception:
        return {"gaps": [], "graph": graph}


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

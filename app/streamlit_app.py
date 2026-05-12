"""Streamlit UI for Research Trail Builder."""

from __future__ import annotations

import contextvars
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

import streamlit as st

from research_trail.agents.nodes import (
    build_graph as node_build_graph,
    identify_gaps as node_identify_gaps,
    scope_query as node_scope_query,
    screen_and_extract as node_screen_and_extract,
    search_papers as node_search_papers,
    synthesize as node_synthesize,
)
from research_trail.config import PROJECT_ROOT, get_settings
from research_trail.runlog import (
    _current_handler,
    _current_node,
    get_current_handler,
    open_run,
    serialize_state,
    write_state,
)

st.set_page_config(page_title="Research Trail Builder", layout="wide")
st.title("Research Trail Builder")
st.caption("Graph-based knowledge maps of scientific literature.")

# ── session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("result", None),
    ("annotations", {}),
    ("selected_node", None),
    ("eval_result", None),
    ("llm_handler", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


@contextmanager
def _attribute_to(node_name: str):
    """Bind the active run's LLM callback handler and node name for post-run actions.

    During the pipeline run the @node decorator + open_run set these contextvars,
    but Streamlit buttons (Evaluate, Expand topic) fire on later script reruns
    after open_run has exited, so without this re-bind their LLM calls would
    have no callback attached at all — token/cost would silently disappear.
    The handler is the same instance the run created, fetched from session_state.
    """
    handler = st.session_state.get("llm_handler")
    h_tok = _current_handler.set(handler) if handler is not None else None
    n_tok = _current_node.set(node_name)
    try:
        yield handler
    finally:
        _current_node.reset(n_tok)
        if h_tok is not None:
            _current_handler.reset(h_tok)

settings = get_settings()
if settings.offline:
    st.warning(
        "Offline mode: OPENAI_API_KEY is not set — results are deterministic stubs."
    )

# ── query input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Research query",
    placeholder="e.g. graph neural networks for drug discovery",
)
run = st.button("Run", type="primary", disabled=not query)

# ── graph helpers ─────────────────────────────────────────────────────────────
_NODE_COLOR = {
    "paper": "#4a90d9",
    "claim": "#27ae60",
    "method": "#e67e22",
    "gap": "#e74c3c",
}


def _build_paper_index(g: dict, papers: list) -> dict[str, tuple[int, Any]]:
    """Map paper node ID → (1-based display index, Paper object)."""
    lookup = {f"paper:{p.id}": p for p in papers}
    index: dict[str, tuple[int, Any]] = {}
    n = 0
    for node in g.get("nodes", []):
        nid = node["id"]
        if node.get("kind") == "paper" and nid in lookup:
            n += 1
            index[nid] = (n, lookup[nid])
    return index


def _build_gap_index(g: dict) -> dict[str, tuple[int, str]]:
    """Map gap node ID → (1-based display index, gap text)."""
    index: dict[str, tuple[int, str]] = {}
    n = 0
    for node in g.get("nodes", []):
        if node.get("kind") == "gap":
            n += 1
            index[node["id"]] = (n, node.get("label", node["id"]))
    return index


def _annotate_summary_with_paper_refs(summary: str, paper_index: dict[str, tuple[int, Any]]) -> str:
    """Append clickable [P#] references after paper titles in synthesis text."""
    if not summary or not paper_index:
        return summary

    title_refs = sorted(
        (
            (paper.title, idx, getattr(paper, "url", ""))
            for idx, paper in paper_index.values()
            if getattr(paper, "title", "")
        ),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    annotated = summary
    for title, idx, url in title_refs:
        pattern = re.compile(rf"{re.escape(title)}(?!\s*\[P\d+\])")
        ref = f"[\\[P{idx}\\]]({url})" if url else f"\\[P{idx}\\]"
        annotated = pattern.sub(lambda m: f"{m.group(0)} {ref}", annotated)
    return annotated


def _render_graph_interactive(
    g: dict,
    annotations: dict,
    paper_index: dict,
    gap_index: dict,
) -> str | None:
    """Render the concept graph and return the clicked node ID (or None)."""
    try:
        from streamlit_agraph import Config, Edge, Node, agraph

        # Rank concept nodes by how many papers reference them; keep top 30.
        concept_paper_count: dict[str, int] = {}
        for e in g.get("edges", []):
            if e.get("source", "").startswith("paper:"):
                concept_paper_count[e.get("target", "")] = (
                    concept_paper_count.get(e.get("target", ""), 0) + 1
                )
        top_concepts: set[str] = set(
            nid for nid, _ in sorted(concept_paper_count.items(), key=lambda x: -x[1])[:30]
        )

        nodes = []
        for n in g.get("nodes", []):
            kind = n.get("kind", "")
            nid = n["id"]

            if kind == "gap":
                continue  # gaps shown in text expander, not graph
            if kind in ("claim", "method") and nid not in top_concepts:
                continue

            if kind == "paper" and nid in paper_index:
                idx, paper = paper_index[nid]
                # title = URL so double-click opens the paper; hover shows the URL.
                # (streamlit-agraph uses title's innerHTML as the double-click navigation target)
                paper_url = paper.url if (paper.url and paper.url.startswith("http")) else None
                nodes.append(
                    Node(
                        id=nid,
                        label=f"P{idx}",
                        shape="circle",
                        color=_NODE_COLOR["paper"],
                        size=28,
                        title=paper_url,
                        font={"color": "white", "size": 12, "bold": True},
                    )
                )
                continue

            raw_label = n.get("label", "")
            nodes.append(
                Node(
                    id=nid,
                    label="",
                    shape="circle",
                    color=_NODE_COLOR.get(kind, "#aaaaaa"),
                    size=14,
                    font={"color": "white", "size": 12, "bold": True},
                )
            )

        # Only draw edges whose target concept node is actually being rendered.
        rendered_ids = {n.id for n in nodes}
        edges = [
            Edge(source=e["source"], target=e["target"], label=e.get("relation", ""))
            for e in g.get("edges", [])
            if e["source"] in rendered_ids and e["target"] in rendered_ids
        ]
        config = Config(
            width="100%",
            height=500,
            directed=True,
            physics=True,
            solver="repulsion",
            repulsion={
                "nodeDistance": 100,
                "springLength": 120,
                "damping": 0.9,
            },
        )
        return agraph(nodes=nodes, edges=edges, config=config)
    except Exception as exc:
        st.json(g)
        st.caption(f"(graph viz fallback: {exc})")
        return None


def _expand_node(concept_label: str, result: dict) -> None:
    """Search for papers on concept_label and merge into the existing graph."""
    if settings.offline:
        st.info("Expansion is unavailable in offline mode.")
        return

    from research_trail.extraction.extractor import extract_from_paper
    from research_trail.graph.builder import build_concept_graph, enrich_graph_with_gaps
    from research_trail.search.aggregator import search_all

    with st.spinner(f"Expanding '{concept_label}'…"):
        new_papers = search_all(concept_label, per_source_limit=3)
        if not new_papers:
            st.info("No new papers found for that concept.")
            return

        existing_ids = {p.id for p in result.get("papers", [])}
        fresh = [p for p in new_papers if p.id not in existing_ids]
        if not fresh:
            st.info("All found papers are already in the graph.")
            return

        workers = min(settings.extract_concurrency, len(fresh))
        # Re-bind the run's LLM handler + node name so extractions kicked off
        # by this button still land in messages.jsonl and the cost accumulator.
        with _attribute_to("expand_topic"):
            ctx = contextvars.copy_context()

            def _run(paper):
                return ctx.run(extract_from_paper, paper)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                new_extractions = list(pool.map(_run, fresh))

        result["papers"] = result.get("papers", []) + fresh
        all_extractions = result.get("extractions", []) + new_extractions
        result["extractions"] = all_extractions
        result["graph"] = enrich_graph_with_gaps(
            build_concept_graph(all_extractions),
            result.get("gaps", []),
        )
        st.session_state.result = result


def _render_node_inspector(
    node_id: str,
    g: dict,
    result: dict,
    paper_index: dict,
    gap_index: dict,
) -> None:
    """Show metadata, annotation editor, and expand button for a selected node."""
    node_meta = next((n for n in g.get("nodes", []) if n["id"] == node_id), None)
    if not node_meta:
        return

    kind = node_meta.get("kind", "unknown")

    with st.container(border=True):
        if kind == "paper" and node_id in paper_index:
            idx, paper = paper_index[node_id]
            st.markdown(f"**P{idx} · {paper.title}**")
            st.caption(
                f"{', '.join(paper.authors[:3])} · {paper.year or 'n/a'} · {paper.source}"
            )
            if paper.abstract:
                with st.expander("Abstract"):
                    st.write(paper.abstract)
            if paper.url:
                st.markdown(f"[Open paper →]({paper.url})")

        elif kind == "gap" and node_id in gap_index:
            idx, gap_text = gap_index[node_id]
            st.markdown(f"**G{idx} · Research gap**")
            st.write(gap_text)

        else:
            label = node_meta.get("label", node_id)
            kind_badge = {"claim": "💬 Claim", "method": "🔧 Method"}.get(kind, kind)
            st.markdown(f"**{kind_badge}:** {label}")

        annotation_key = f"ann_{node_id}"
        current_note = st.session_state.annotations.get(node_id, "")
        note = st.text_area(
            "Your notes",
            value=current_note,
            key=annotation_key,
            height=80,
            placeholder="Add a note about this node…",
        )
        col_save, col_expand = st.columns([1, 3])
        with col_save:
            if st.button("Save note", key=f"save_{node_id}"):
                st.session_state.annotations[node_id] = note
                st.success("Saved.")
        with col_expand:
            if kind in ("claim", "method", "gap"):
                label = node_meta.get("label", node_id)
                if st.button("Expand topic", key=f"expand_{node_id}", type="primary"):
                    _expand_node(label, st.session_state.result)
                    st.rerun()


# ── pipeline execution ────────────────────────────────────────────────────────
def _render_step_details(name: str, state: dict) -> None:
    """Render verbose per-step detail inside the step's expander body."""
    if name == "scope_query":
        for i, s in enumerate(state.get("sub_problems", []), 1):
            st.markdown(f"{i}. {s}")

    elif name == "search_papers":
        papers = state.get("papers", [])
        if papers:
            src = Counter(p.source for p in papers)
            st.caption(" · ".join(f"**{s}**: {c}" for s, c in sorted(src.items())))
        for p in papers:
            title = f"[{p.title}]({p.url})" if p.url else p.title
            authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            st.markdown(f"- {title}  *({authors}, {p.year or 'n/a'}, `{p.source}`)*")

    elif name == "screen_and_extract":
        for ext in state.get("extractions", []):
            st.markdown(
                f"**{ext.paper_id}** · conf {ext.confidence:.2f} · "
                f"{len(ext.claims)} claims · {len(ext.methods)} methods · "
                f"{len(ext.evidence)} evidence"
            )

    elif name == "build_graph":
        g = state.get("graph", {})
        nodes = g.get("nodes", [])
        kinds = Counter(n.get("kind") for n in nodes)
        st.caption(" · ".join(f"**{k}**: {v}" for k, v in sorted(kinds.items())))

    elif name == "synthesize":
        summary = state.get("summary", "") or "(empty)"
        st.write(summary)

    elif name == "identify_gaps":
        for i, gap in enumerate(state.get("gaps", []), 1):
            st.markdown(f"{i}. {gap}")


def _step_summary(name: str, state: dict) -> str:
    if name == "scope_query":
        return f"{len(state.get('sub_problems', []))} sub-problems"
    if name == "search_papers":
        return f"{len(state.get('papers', []))} papers"
    if name == "screen_and_extract":
        return f"{len(state.get('extractions', []))} extractions"
    if name == "build_graph":
        g = state.get("graph", {})
        return f"{len(g.get('nodes', []))} nodes, {len(g.get('edges', []))} edges"
    if name == "synthesize":
        return f"{len((state.get('summary') or '').split())} words"
    if name == "identify_gaps":
        return f"{len(state.get('gaps', []))} gaps"
    return ""


def _format_cost(cost: float, unknown: bool) -> str:
    if unknown:
        return "$? (model untracked)"
    return f"${cost:.4f}"


def _node_usage(name: str) -> dict | None:
    """Look up tokens/cost for a node from the active runlog handler."""
    handler = get_current_handler()
    if handler is None:
        return None
    return handler.totals_by_node().get(name)


def _render_step(name: str, elapsed: float, state: dict) -> None:
    summary = _step_summary(name, state)
    usage = _node_usage(name)
    cost_part = ""
    if usage:
        cost_part = (
            f" · {usage['total_tokens']:,} tok · "
            f"{_format_cost(usage['cost_usd'], False)}"
        )
    label = f"✓ `{name}` ({elapsed:.1f}s{cost_part}) — {summary}"
    st.markdown(label)
    with st.container(border=True):
        if usage:
            st.caption(
                f"**{usage['calls']} call(s)** · "
                f"prompt: {usage['prompt_tokens']:,} tok · "
                f"completion: {usage['completion_tokens']:,} tok · "
                f"cost: {_format_cost(usage['cost_usd'], False)}"
            )
        _render_step_details(name, state)


def _run_node_with_timer(name: str, fn, state: dict) -> float:
    """Run a node in a worker thread while ticking a live elapsed-time line.

    runlog uses contextvars for the per-run logging handler and current node
    name, so we hand the worker thread a copy of the current context — a
    plain ``Thread(target=fn)`` would lose that and silently skip log lines.
    The line also surfaces tokens/cost for the active node so the user can
    see spend accumulate in real time once the LLM call returns.
    """
    t0 = time.perf_counter()
    placeholder = st.empty()
    result_box: dict[str, Any] = {}
    ctx = contextvars.copy_context()

    def _runner() -> None:
        try:
            partial = fn(state)
            result_box["partial"] = partial if isinstance(partial, dict) else {}
        except Exception as exc:  # surfaced after join so the UI stays consistent
            result_box["error"] = exc

    th = threading.Thread(target=ctx.run, args=(_runner,), daemon=True)
    th.start()
    while th.is_alive():
        elapsed = time.perf_counter() - t0
        usage = _node_usage(name)
        suffix = ""
        if usage and usage["total_tokens"]:
            suffix = (
                f" · {usage['total_tokens']:,} tok · "
                f"{_format_cost(usage['cost_usd'], False)}"
            )
        placeholder.markdown(f"⏳ `{name}` _running… {elapsed:.1f}s{suffix}_")
        time.sleep(0.2)
    th.join()
    placeholder.empty()
    if "error" in result_box:
        raise result_box["error"]
    state.update(result_box.get("partial", {}))
    return time.perf_counter() - t0


def _extract_with_progress(papers: list) -> list:
    """Run per-paper extraction concurrently with a live progress bar.

    Bypasses screen_and_extract's internal pool so the UI can surface
    per-paper completions. Order is preserved: ``extractions[i]`` corresponds
    to ``papers[i]``. The node-name contextvar is set explicitly here (since
    we skip the @node wrapper) and copy_context is used so worker threads
    see the same value — without this, every per-paper LLM call would log
    with node=None and the screen_and_extract step expander would show $0.
    """
    from research_trail.extraction.extractor import extract_from_paper

    n = len(papers)
    workers = min(get_settings().extract_concurrency, n)
    extractions: list = [None] * n
    bar = st.progress(0.0, text=f"Extracting 0/{n} papers…")

    n_tok = _current_node.set("screen_and_extract")
    try:
        # Each worker needs its own Context copy: a single shared ctx object can
        # only be entered by one thread at a time, so concurrent ctx.run() calls
        # on the same object race and crash. Copying once per paper in the main
        # thread (where _current_node is already set) gives each worker an
        # independent context with the correct node name.
        def _run(paper: Any, ctx: contextvars.Context) -> Any:
            return ctx.run(extract_from_paper, paper)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_run, p, contextvars.copy_context()): i
                for i, p in enumerate(papers)
            }
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                extractions[idx] = fut.result()
                done += 1
                paper_title = papers[idx].title[:50]
                bar.progress(
                    done / n,
                    text=f"Extracting {done}/{n} · just finished: {paper_title}…",
                )
    finally:
        _current_node.reset(n_tok)
    bar.empty()
    return extractions


if run and query:
    st.session_state.result = None
    st.session_state.selected_node = None
    st.session_state.eval_result = None

    pipeline_t0 = time.perf_counter()
    with open_run(query) as run_dir:
        with st.status("Running pipeline…", expanded=True) as status:
            state_view: dict = {"query": query}

            def _step(name: str, fn) -> None:
                elapsed = _run_node_with_timer(name, fn, state_view)
                _render_step(name, elapsed, state_view)

            _step("scope_query", node_scope_query)
            _step("search_papers", node_search_papers)

            papers = state_view.get("papers", [])
            if get_settings().offline or not papers:
                _step("screen_and_extract", node_screen_and_extract)
            else:
                t0 = time.perf_counter()
                state_view["extractions"] = _extract_with_progress(papers)
                _render_step("screen_and_extract", time.perf_counter() - t0, state_view)

            _step("build_graph", node_build_graph)
            _step("synthesize", node_synthesize)
            _step("identify_gaps", node_identify_gaps)

            status.update(label="Pipeline complete", state="complete", expanded=False)

        run_elapsed = time.perf_counter() - pipeline_t0
        # Stash the handler + elapsed time in session_state so post-run buttons
        # (Evaluate, Expand topic) can re-bind it via _attribute_to and keep
        # accumulating, and so the cost banner re-renders on every script rerun.
        st.session_state.llm_handler = get_current_handler()
        st.session_state.run_elapsed = run_elapsed

        write_state(run_dir, serialize_state(query, state_view))
        try:
            rel = run_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = run_dir
        st.caption(f"Artifacts saved to `{rel}`")

    st.session_state.result = state_view


def _render_cost_banner() -> None:
    """Render the tokens/cost/time banner from the live handler.

    Shown on every rerun (not just immediately after the pipeline) so post-run
    LLM calls — Evaluate, Expand topic — surface their additional cost as the
    user clicks. Reads totals fresh each time so the banner stays current.
    """
    handler = st.session_state.get("llm_handler")
    if handler is None:
        return
    totals = handler.totals()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total tokens", f"{totals['total_tokens']:,}")
    c2.metric(
        "Total cost",
        _format_cost(totals["cost_usd"], totals.get("cost_unknown_model", False)),
    )
    c3.metric("Pipeline time", f"{st.session_state.get('run_elapsed', 0):.1f}s")


_render_cost_banner()

# ── results rendering ─────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result:
    papers = result.get("papers", [])
    g = result.get("graph", {})
    paper_index = _build_paper_index(g, papers) if g else {}
    gap_index = _build_gap_index(g) if g else {}

    summary = result.get("summary", "")
    if summary:
        st.subheader("Synthesis")
        st.markdown(_annotate_summary_with_paper_refs(summary, paper_index))

    gaps = result.get("gaps", [])
    if gaps:
        with st.expander("Research Gaps", expanded=True):
            for i, gap in enumerate(gaps, 1):
                st.markdown(f"**{i}.** {gap}")

    # Evaluate + download row
    col_eval, col_dl, _ = st.columns([1, 1, 3])
    with col_eval:
        if st.button("Evaluate with LLM judge"):
            from research_trail.evaluation.llm_judge import judge

            with st.spinner("Running LLM judge…"), _attribute_to("evaluate"):
                g_ref = result.get("graph", {})
                rubric = judge(
                    result.get("query", query),
                    {
                        "summary": summary,
                        "gaps": gaps,
                        "graph": {
                            "nodes": len(g_ref.get("nodes", [])),
                            "edges": len(g_ref.get("edges", [])),
                        },
                    },
                )
                st.session_state.eval_result = rubric

    with col_dl:
        graph_json = json.dumps(result.get("graph", {}), indent=2, default=str)
        st.download_button(
            "Download graph JSON",
            data=graph_json,
            file_name="concept_graph.json",
            mime="application/json",
        )

    eval_result = st.session_state.get("eval_result")
    if eval_result:
        score = eval_result.score
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Relevance", f"{score.relevance}/5")
        c2.metric("Coverage", f"{score.coverage}/5")
        c3.metric("Structure", f"{score.structural_organization}/5")
        c4.metric("Insightfulness", f"{score.insightfulness}/5")
        c5.metric("Overall", f"{score.mean:.1f}/5")
        if score.rationale:
            st.caption(f"Judge rationale: {score.rationale}")

    if g and g.get("nodes"):
        st.subheader("Concept Graph")

        # Legend
        legend_cols = st.columns(3)
        for col, (color, label) in zip(
            legend_cols,
            [
                (_NODE_COLOR["paper"], "= Paper"),
                (_NODE_COLOR["claim"], "= Claim"),
                (_NODE_COLOR["method"], "= Method"),
            ],
        ):
            col.markdown(
                f'<span style="color:{color}; font-size:1.2em">⬤</span> {label}',
                unsafe_allow_html=True,
            )

        graph_col, details_col = st.columns([6, 4])
        with graph_col:
            with st.container(border=True):
                st.caption("Papers = P1, P2… · Gaps = G1, G2… · Hover any node for full text · Click to inspect · Claim/method/gap nodes can be expanded")
                clicked = _render_graph_interactive(g, st.session_state.annotations, paper_index, gap_index)
                if clicked:
                    st.session_state.selected_node = clicked
        with details_col:
            st.markdown("**Node details**")
            sel = st.session_state.selected_node
            if sel:
                _render_node_inspector(sel, g, result, paper_index, gap_index)
            else:
                st.info("Click a node in the graph to inspect details.")

        # Paper index table
        # if paper_index:
        #     st.markdown("**Paper index**")
        #     for nid, (idx, paper) in sorted(paper_index.items(), key=lambda x: x[1][0]):
        #         link = f"[{paper.title}]({paper.url})" if paper.url else paper.title
        #         authors = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")
        #         st.markdown(f"**P{idx}** · {link} · {authors} · {paper.year or 'n/a'} · `{paper.source}`")

        # Gap index table
        if gap_index:
            st.subheader("Gap index")
            for nid, (idx, gap_text) in sorted(gap_index.items(), key=lambda x: x[1][0]):
                st.markdown(f"**G{idx}** · {gap_text}")

    
    st.subheader("Paper Index and Extractions")
    # Merged Papers + Extractions view
    extractions = result.get("extractions", [])
    if not papers:
        st.info("No papers retrieved.")
    elif not extractions:
        st.info("No extractions available.")
    else:
        # Build a mapping from paper id to (index, Paper object)
        paper_display_index = {p.id: (idx, p) for idx, (nid, (idx, p)) in enumerate(sorted(paper_index.items(), key=lambda x: x[1][0]), 1)}
        # Map paper_id → first extraction (by order in extractions)
        paper_to_extraction = {}
        for ext in extractions:
            paper_to_extraction.setdefault(ext.paper_id, ext)
        for nid, (idx, paper) in sorted(paper_index.items(), key=lambda x: x[1][0]):
            ext = paper_to_extraction.get(paper.id)
            reference = f"**[P{idx}]**"
            title_md = paper.title
            conf_pct = f"{ext.confidence * 100:.0f}%" if ext and hasattr(ext, "confidence") else "n/a"
            expander_label = f"{reference} {paper.title}  (confidence: {conf_pct})"
            with st.expander(expander_label):
                st.markdown(
                    f"**URL**: {paper.url}<br>"
                    f"**Authors**: {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}<br>"
                    f"**Year**: {paper.year or 'n/a'} &nbsp; | &nbsp; **Source**: `{paper.source}`",
                    unsafe_allow_html=True,
                )
           
                if ext:
                    if ext.claims:
                        st.markdown("**Claims:**")
                        for c in ext.claims:
                            st.markdown(f"- {c}")
                    if ext.methods:
                        st.markdown("**Methods:**")
                        for m in ext.methods:
                            st.markdown(f"- {m}")
                    if ext.evidence:
                        st.markdown("**Evidence:**")
                        for e in ext.evidence:
                            st.markdown(f"- {e}")
                else:
                    st.info("No extraction available for this paper.")

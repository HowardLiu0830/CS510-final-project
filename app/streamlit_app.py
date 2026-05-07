"""Streamlit UI for Research Trail Builder."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from research_trail.agents.graph import compile_graph
from research_trail.config import PROJECT_ROOT, get_settings
from research_trail.runlog import open_run, serialize_state, write_state

st.set_page_config(page_title="Research Trail Builder", layout="wide")
st.title("Research Trail Builder")
st.caption("Graph-based knowledge maps of scientific literature.")

# ── session state ─────────────────────────────────────────────────────────────
for _key, _default in [("result", None), ("annotations", {}), ("selected_node", None)]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

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
_NODE_COLOR = {"paper": "#4a90d9", "claim": "#27ae60", "method": "#e67e22"}
_NODE_SIZE = {"paper": 25, "claim": 12, "method": 12}


def _render_graph_interactive(g: dict, annotations: dict) -> str | None:
    """Render the concept graph and return the clicked node ID (or None)."""
    try:
        from streamlit_agraph import Config, Edge, Node, agraph

        nodes = [
            Node(
                id=n["id"],
                label=n.get("label", n["id"])[:40],
                color=_NODE_COLOR.get(n.get("kind", ""), "#aaaaaa"),
                size=_NODE_SIZE.get(n.get("kind", ""), 12),
                title=annotations.get(n["id"], ""),  # shown as tooltip
            )
            for n in g.get("nodes", [])
        ]
        edges = [
            Edge(source=e["source"], target=e["target"], label=e.get("relation", ""))
            for e in g.get("edges", [])
        ]
        config = Config(width="100%", height=500, directed=True, physics=True)
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
    from research_trail.graph.builder import build_concept_graph
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
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_extractions = list(pool.map(extract_from_paper, fresh))

        result["papers"] = result.get("papers", []) + fresh
        all_extractions = result.get("extractions", []) + new_extractions
        result["extractions"] = all_extractions
        result["graph"] = build_concept_graph(all_extractions)
        st.session_state.result = result


def _render_node_inspector(node_id: str, g: dict, result: dict) -> None:
    """Show metadata, annotation editor, and expand button for a selected node."""
    node_meta = next((n for n in g.get("nodes", []) if n["id"] == node_id), None)
    if not node_meta:
        return

    kind = node_meta.get("kind", "unknown")
    label = node_meta.get("label", node_id)

    with st.container(border=True):
        st.markdown(f"**[{kind}]** {label}")

        if kind == "paper":
            paper_id = node_id.removeprefix("paper:")
            paper = next((p for p in result.get("papers", []) if p.id == paper_id), None)
            if paper:
                title_md = f"[{paper.title}]({paper.url})" if paper.url else paper.title
                st.markdown(title_md)
                st.caption(
                    f"{', '.join(paper.authors[:3])} · {paper.year or 'n/a'} · {paper.source}"
                )
                if paper.abstract:
                    with st.expander("Abstract"):
                        st.write(paper.abstract)

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
            if kind in ("claim", "method"):
                if st.button("Expand topic", key=f"expand_{node_id}", type="primary"):
                    _expand_node(label, st.session_state.result)
                    st.rerun()


# ── pipeline execution ────────────────────────────────────────────────────────
if run and query:
    st.session_state.result = None
    st.session_state.selected_node = None

    with open_run(query) as run_dir:
        with st.status("Running pipeline…", expanded=True) as status:
            pipeline = compile_graph()
            state_view: dict = {}
            t0 = time.perf_counter()
            for update in pipeline.stream({"query": query}):
                for node_name, partial in update.items():
                    if not isinstance(partial, dict):
                        continue
                    state_view.update(partial)
                    dt = time.perf_counter() - t0
                    status.write(f"✓ `{node_name}` ({dt:.1f}s)")
                    t0 = time.perf_counter()
            status.update(label="Pipeline complete", state="complete", expanded=False)

        write_state(run_dir, serialize_state(query, state_view))
        try:
            rel = run_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = run_dir
        st.caption(f"Artifacts saved to `{rel}`")

    st.session_state.result = state_view

# ── results rendering ─────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result:
    summary = result.get("summary", "")
    if summary:
        st.subheader("Synthesis")
        st.write(summary)

    gaps = result.get("gaps", [])
    if gaps:
        with st.expander("Research Gaps", expanded=True):
            for i, gap in enumerate(gaps, 1):
                st.markdown(f"**{i}.** {gap}")

    g = result.get("graph", {})
    if g and g.get("nodes"):
        st.subheader("Concept Graph")
        st.caption(
            "Click a node to inspect it or add notes. "
            "Claim/method nodes have an **Expand topic** button to fetch more papers."
        )
        clicked = _render_graph_interactive(g, st.session_state.annotations)
        if clicked:
            st.session_state.selected_node = clicked

        sel = st.session_state.selected_node
        if sel:
            _render_node_inspector(sel, g, result)

    tab_papers, tab_extractions = st.tabs(["Papers", "Extractions"])
    with tab_papers:
        papers = result.get("papers", [])
        if papers:
            for p in papers:
                title_md = f"[{p.title}]({p.url})" if p.url else p.title
                st.markdown(
                    f"**{title_md}** · {', '.join(p.authors[:3])} · "
                    f"{p.year or 'n/a'} · `{p.source}`"
                )
        else:
            st.info("No papers retrieved.")

    with tab_extractions:
        extractions = result.get("extractions", [])
        if extractions:
            for ext in extractions:
                conf_pct = f"{ext.confidence * 100:.0f}%"
                with st.expander(f"{ext.paper_id}  (confidence: {conf_pct})"):
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
            st.info("No extractions available.")

"""Streamlit UI for Research Trail Builder."""

from __future__ import annotations

import time

import streamlit as st

from research_trail.agents.graph import compile_graph
from research_trail.config import PROJECT_ROOT, get_settings
from research_trail.runlog import open_run, serialize_state, write_state

st.set_page_config(page_title="Research Trail Builder", layout="wide")
st.title("Research Trail Builder")
st.caption("Graph-based knowledge maps of scientific literature.")

settings = get_settings()
if settings.offline:
    st.warning(
        "Offline mode: OPENAI_API_KEY is not set, so results are deterministic stubs."
    )

query = st.text_input(
    "Research query",
    placeholder="e.g. graph neural networks for drug discovery",
)
run = st.button("Run", type="primary", disabled=not query)


def _render_graph(container, g: dict) -> None:
    with container.container():
        try:
            from streamlit_agraph import Config, Edge, Node, agraph

            nodes = [Node(id=n["id"], label=n.get("label", n["id"])) for n in g.get("nodes", [])]
            edges = [
                Edge(source=e["source"], target=e["target"], label=e.get("relation", ""))
                for e in g.get("edges", [])
            ]
            agraph(
                nodes=nodes,
                edges=edges,
                config=Config(width=700, height=500, directed=True),
            )
        except Exception as exc:  # pragma: no cover
            st.json(g)
            st.info(f"(graph viz fallback: {exc})")


if run:
    status = st.status("Running pipeline…", expanded=True)
    artifacts_caption = st.empty()

    synthesis_ph = st.empty()
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Sub-problems")
        sub_ph = st.empty()
        st.subheader("Papers")
        papers_ph = st.empty()
        st.subheader("Extractions")
        extractions_ph = st.empty()
    with cols[1]:
        st.subheader("Concept graph")
        graph_ph = st.empty()

    state_view: dict = {}
    with open_run(query) as run_dir:
        graph = compile_graph()
        node_t0 = time.perf_counter()
        for update in graph.stream({"query": query}):
            for node_name, partial in update.items():
                if not isinstance(partial, dict):
                    continue
                state_view.update(partial)
                dt = time.perf_counter() - node_t0
                status.write(f"✓ `{node_name}` ({dt:.1f}s)")
                node_t0 = time.perf_counter()

                if node_name == "scope_query":
                    sub_ph.write(partial.get("sub_problems", []))
                elif node_name == "search_papers":
                    papers_ph.write([p.model_dump() for p in partial.get("papers", [])])
                elif node_name == "screen_and_extract":
                    extractions_ph.write(
                        [e.model_dump() for e in partial.get("extractions", [])]
                    )
                elif node_name == "build_graph":
                    _render_graph(graph_ph, partial.get("graph", {"nodes": [], "edges": []}))
                elif node_name == "synthesize":
                    summary = partial.get("summary", "")
                    if summary:
                        with synthesis_ph.container():
                            st.subheader("Synthesis")
                            st.write(summary)

        write_state(run_dir, serialize_state(query, state_view))

    status.update(label="Pipeline complete", state="complete", expanded=False)
    try:
        rel = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = run_dir
    artifacts_caption.caption(f"Run artifacts saved to `{rel}`")

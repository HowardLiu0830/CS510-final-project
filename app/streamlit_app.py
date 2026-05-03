"""Streamlit UI for Research Trail Builder."""

from __future__ import annotations

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

if run:
    with st.spinner("Running pipeline…"):
        with open_run(query) as run_dir:
            graph = compile_graph()
            final_state = graph.invoke({"query": query})
            write_state(run_dir, serialize_state(query, final_state))
    try:
        rel = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = run_dir
    st.caption(f"Run artifacts saved to `{rel}`")

    summary = final_state.get("summary")
    if summary:
        st.subheader("Synthesis")
        st.write(summary)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Sub-problems")
        st.write(final_state.get("sub_problems", []))
        st.subheader("Papers")
        st.write([p.model_dump() for p in final_state.get("papers", [])])
        st.subheader("Extractions")
        st.write([e.model_dump() for e in final_state.get("extractions", [])])
    with cols[1]:
        st.subheader("Concept graph")
        g = final_state.get("graph", {"nodes": [], "edges": []})
        try:
            from streamlit_agraph import Config, Edge, Node, agraph

            nodes = [Node(id=n["id"], label=n.get("label", n["id"])) for n in g["nodes"]]
            edges = [
                Edge(source=e["source"], target=e["target"], label=e.get("relation", ""))
                for e in g["edges"]
            ]
            agraph(
                nodes=nodes,
                edges=edges,
                config=Config(width=700, height=500, directed=True),
            )
        except Exception as exc:  # pragma: no cover
            st.json(g)
            st.info(f"(graph viz fallback: {exc})")

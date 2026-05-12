"""Evaluation baselines for the research-trail pipeline.

Two ablations are exposed:

* ``zero_shot``: the LLM answers the query directly, with no retrieval,
  no extraction, and no graph. Establishes a floor for how much the
  retrieval-and-structure pipeline buys over a single prompt.
* ``no_graph``: the full retrieval pipeline (scope, search, extract,
  synthesize, identify gaps) runs, but the concept graph is *not* built
  and is *not* fed into synthesis or gap-finding. Isolates the
  contribution of the graph specifically.

Both baselines return a state dict with the same schema as the main
pipeline so ``runs_to_jsonl`` and ``run_eval`` consume them unchanged.
"""

from research_trail.baselines.no_graph import run_no_graph
from research_trail.baselines.zero_shot import run_zero_shot

__all__ = ["run_no_graph", "run_zero_shot"]

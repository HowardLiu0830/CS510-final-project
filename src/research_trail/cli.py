"""Console-script entry points (registered in pyproject.toml)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _final_state_issues(state: dict) -> list[str]:
    """Return human-readable issues with a final pipeline state, or [] if healthy.

    Used by ``run_agent`` to fail loud when the graph produces empty intermediate
    state — otherwise eval would silently score zeros against missing data.
    """
    issues: list[str] = []
    if not state.get("sub_problems"):
        issues.append("no sub-problems generated (scope_query produced nothing)")
    if not state.get("papers"):
        issues.append("no papers retrieved (all search sources failed or returned empty)")
    if not state.get("summary"):
        issues.append("empty synthesis summary")
    return issues


def run_agent(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the research-trail agent.")
    parser.add_argument("query", help="Research query / topic")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the generation model for this run "
            "(e.g. 'moonshotai/kimi-k2.6', 'google/gemini-3-flash-preview'). "
            "Does NOT affect the judge model, which still reads OPENAI_MODEL."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional path to write a single JSON result"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the auto-created per-run directory (default: data/runs/<ts>__<slug>/)",
    )
    parser.add_argument(
        "--no-run-dir",
        action="store_true",
        help="Disable per-run persistence (no log/state files written)",
    )
    args = parser.parse_args(argv)

    from research_trail.agents.graph import compile_graph
    from research_trail.llm.client import override_model
    from research_trail.runlog import open_run, serialize_state, write_state

    graph = compile_graph()
    # Tag the query so model-conditioned runs are distinguishable in eval.
    tagged_query = f"[model:{args.model}] {args.query}" if args.model else args.query

    # override_model must wrap open_run so _write_meta (in open_run's
    # finally) still sees the override when it records the effective model.
    if args.no_run_dir:
        with override_model(args.model):
            state = graph.invoke({"query": args.query})
            serial = serialize_state(tagged_query, state)
        run_dir_used: Path | None = None
    else:
        with override_model(args.model):
            with open_run(tagged_query, run_dir=args.run_dir) as run_dir_used:
                state = graph.invoke({"query": args.query})
                serial = serialize_state(tagged_query, state)
                write_state(run_dir_used, serial)

    text = json.dumps(serial, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if run_dir_used is not None:
        print(f"\n# run artifacts: {run_dir_used}", file=sys.stderr)

    issues = _final_state_issues(serial)
    if issues:
        print("WARNING: pipeline produced incomplete output:", file=sys.stderr)
        for msg in issues:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    return 0


def run_baseline(argv: list[str] | None = None) -> int:
    """Run one of the evaluation baselines and persist its output like a normal run.

    The output schema matches the main pipeline so ``runs_to_jsonl`` + ``run_eval``
    consume baseline runs unchanged.
    """
    parser = argparse.ArgumentParser(description="Run an evaluation baseline.")
    parser.add_argument(
        "--kind",
        choices=["zero_shot", "no_graph"],
        required=True,
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the generation model for this baseline run. "
            "Judge model is unaffected."
        ),
    )
    parser.add_argument("query", help="Research query / topic")
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional path to write a single JSON result"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the auto-created per-run directory.",
    )
    parser.add_argument(
        "--no-run-dir",
        action="store_true",
        help="Disable per-run persistence (no log/state files written)",
    )
    args = parser.parse_args(argv)

    from research_trail.baselines import run_no_graph, run_zero_shot
    from research_trail.llm.client import override_model
    from research_trail.runlog import open_run, serialize_state, write_state

    runner = run_zero_shot if args.kind == "zero_shot" else run_no_graph
    # Tag the run dir so eval can distinguish baseline runs from full-pipeline runs,
    # and also pin the generation model so model-conditioned ablations are visible.
    model_tag = f"[model:{args.model}] " if args.model else ""
    tagged_query = f"[baseline:{args.kind}] {model_tag}{args.query}"

    # override_model wraps open_run so _write_meta sees the override.
    if args.no_run_dir:
        with override_model(args.model):
            serial = serialize_state(tagged_query, runner(args.query))
        run_dir_used: Path | None = None
    else:
        with override_model(args.model):
            with open_run(tagged_query, run_dir=args.run_dir) as run_dir_used:
                serial = serialize_state(tagged_query, runner(args.query))
                write_state(run_dir_used, serial)

    text = json.dumps(serial, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if run_dir_used is not None:
        print(f"\n# run artifacts: {run_dir_used}", file=sys.stderr)
    return 0


def run_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation.")
    parser.add_argument(
        "--input", type=Path, required=True, help="JSONL of {query, output} records"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON for aggregated scores"
    )
    parser.add_argument(
        "--human-forms",
        type=Path,
        default=None,
        help="Optional directory of filled human-eval JSON forms to merge with LLM judgments",
    )
    args = parser.parse_args(argv)

    from research_trail.evaluation.human_eval import load_human_rubric
    from research_trail.evaluation.llm_judge import judge
    from research_trail.evaluation.metrics import aggregate_scores

    rubrics = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rubrics.append(judge(rec["query"], rec["output"]))

    if args.human_forms:
        if not args.human_forms.is_dir():
            print(f"--human-forms is not a directory: {args.human_forms}", file=sys.stderr)
            return 2
        for p in sorted(args.human_forms.glob("*.json")):
            try:
                rubrics.append(load_human_rubric(p))
            except Exception as exc:
                print(f"skip {p.name}: {exc}", file=sys.stderr)

    summary = aggregate_scores(rubrics)
    payload = {
        "summary": summary,
        "rubrics": [r.model_dump() for r in rubrics],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def runs_to_jsonl(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate per-run state.json files into a JSONL of "
            '{"query", "output"} records that run_eval can consume.'
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory containing run dirs (default: data/runs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: stdout)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help=(
            "Only include runs whose dir-name timestamp is >= this prefix (UTC, ISO-ish). "
            "e.g. '2026-05-03' or '2026-05-03T15-30'."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help=(
            "Include runs whose pipeline produced empty sub_problems/papers/summary. "
            "Default skips them so run_eval doesn't score garbage."
        ),
    )
    args = parser.parse_args(argv)

    from research_trail.config import PROJECT_ROOT

    runs_dir = args.runs_dir or (PROJECT_ROOT / "data" / "runs")
    if not runs_dir.is_dir():
        print(f"runs dir does not exist: {runs_dir}", file=sys.stderr)
        return 2

    since_prefix = args.since.replace(":", "-") if args.since else None

    included: list[str] = []
    skipped_empty = 0
    skipped_old = 0
    skipped_unparseable = 0

    out_lines: list[str] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if since_prefix and run_dir.name < since_prefix:
            skipped_old += 1
            continue
        state_path = run_dir / "state.json"
        if not state_path.exists():
            skipped_unparseable += 1
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip {run_dir.name}: cannot parse state.json: {exc}", file=sys.stderr)
            skipped_unparseable += 1
            continue
        if not args.include_empty and _final_state_issues(state):
            skipped_empty += 1
            continue
        rec = {"query": state.get("query", ""), "output": state}
        out_lines.append(json.dumps(rec, ensure_ascii=False, default=str))
        included.append(run_dir.name)

    text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    print(
        f"runs_to_jsonl: included {len(included)}, "
        f"skipped {skipped_empty} empty, {skipped_old} too-old, "
        f"{skipped_unparseable} missing/unparseable",
        file=sys.stderr,
    )
    return 0


def scaffold_human_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one human-eval JSON form per record in a results JSONL."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="JSONL of {query, output} records"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Directory to write form JSON files"
    )
    args = parser.parse_args(argv)

    from research_trail.evaluation.human_eval import write_form_template

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, line in enumerate(args.input.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        out_path = args.out_dir / f"form_{i:04d}.json"
        write_form_template(rec["query"], rec.get("output", {}), out_path)
        n += 1
    print(f"Wrote {n} form(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(run_agent(sys.argv[1:]))

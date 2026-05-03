"""Aggregation across multiple judges / runs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from research_trail.evaluation.rubric import DIMENSIONS, Rubric


def _per_judge(rubrics: list[Rubric]) -> dict[str, dict[str, Any]]:
    by_judge: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rubrics:
        for d in DIMENSIONS:
            by_judge[r.judge_id][d].append(getattr(r.score, d))

    out: dict[str, dict[str, Any]] = {}
    for jid, dims in by_judge.items():
        n = len(next(iter(dims.values())))
        entry: dict[str, Any] = {"n": n}
        for d, vals in dims.items():
            entry[d] = round(mean(vals), 3)
        entry["overall"] = round(mean([entry[d] for d in DIMENSIONS]), 3)
        out[jid] = entry
    return out


def _inter_rater(rubrics: list[Rubric]) -> dict[str, Any]:
    """Per-dim agreement: for each query reviewed by 2+ judges, compute std and
    pairwise mean-abs-diff across judges, then average across queries."""
    by_query: dict[str, list[Rubric]] = defaultdict(list)
    for r in rubrics:
        by_query[r.query].append(r)

    multi = {q: rs for q, rs in by_query.items() if len({r.judge_id for r in rs}) > 1}
    out: dict[str, Any] = {"queries_with_multiple_judges": len(multi)}
    if not multi:
        return out

    for d in DIMENSIONS:
        stds: list[float] = []
        mads: list[float] = []
        for rs in multi.values():
            vals = [getattr(r.score, d) for r in rs]
            stds.append(pstdev(vals))
            pairs = [
                abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))
            ]
            if pairs:
                mads.append(mean(pairs))
        if stds:
            out[f"{d}_std"] = round(mean(stds), 3)
        if mads:
            out[f"{d}_pairwise_mad"] = round(mean(mads), 3)
    return out


def aggregate_scores(rubrics: list[Rubric]) -> dict[str, Any]:
    """Return per-dimension mean, overall mean, per-judge breakdown, and IRR."""
    if not rubrics:
        return {"n": 0}

    bydim: dict[str, list[int]] = defaultdict(list)
    for r in rubrics:
        for d in DIMENSIONS:
            bydim[d].append(getattr(r.score, d))

    out: dict[str, Any] = {"n": len(rubrics)}
    for d, vals in bydim.items():
        out[d] = round(mean(vals), 3)
    out["overall"] = round(mean([out[d] for d in DIMENSIONS]), 3)
    out["per_judge"] = _per_judge(rubrics)
    out["agreement"] = _inter_rater(rubrics)
    return out

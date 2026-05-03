"""Evaluation harness tests."""

import json

from research_trail.evaluation.metrics import aggregate_scores
from research_trail.evaluation.rubric import Rubric, RubricScore


def _rub(
    judge_id: str, vals: tuple[int, int, int, int], query: str = "q"
) -> Rubric:
    r, c, s, i = vals
    return Rubric(
        judge_id=judge_id,
        query=query,
        score=RubricScore(
            relevance=r,
            coverage=c,
            structural_organization=s,
            insightfulness=i,
        ),
    )


def test_aggregate_empty():
    assert aggregate_scores([]) == {"n": 0}


def test_aggregate_basic():
    rubrics = [
        _rub("llm:a", (4, 4, 4, 4)),
        _rub("human:b", (2, 4, 4, 2)),
    ]
    out = aggregate_scores(rubrics)
    assert out["n"] == 2
    assert out["relevance"] == 3.0
    assert out["coverage"] == 4.0
    assert 2.5 <= out["overall"] <= 4.5


def test_offline_judge_returns_rubric():
    """llm_judge should fall back to a stub rubric when no API key is set."""
    from research_trail.evaluation.llm_judge import judge

    rub = judge("test query", {"papers": []})
    assert 1 <= rub.score.relevance <= 5
    assert rub.score.rationale  # non-empty stub rationale


def test_aggregate_per_judge_breakdown():
    rubrics = [
        _rub("llm:a", (4, 4, 4, 4)),
        _rub("llm:a", (2, 2, 2, 2), query="q2"),
        _rub("human:b", (3, 3, 3, 3)),
    ]
    out = aggregate_scores(rubrics)
    pj = out["per_judge"]
    assert pj["llm:a"]["n"] == 2
    assert pj["llm:a"]["relevance"] == 3.0
    assert pj["llm:a"]["overall"] == 3.0
    assert pj["human:b"]["n"] == 1
    assert pj["human:b"]["overall"] == 3.0


def test_aggregate_inter_rater_agreement():
    # two judges scoring the same query → IRR computable
    rubrics = [
        _rub("llm:a", (5, 5, 5, 5), query="q1"),
        _rub("human:b", (3, 5, 5, 3), query="q1"),
        # one query with only one judge → ignored by IRR
        _rub("llm:a", (4, 4, 4, 4), query="q-solo"),
    ]
    out = aggregate_scores(rubrics)
    agreement = out["agreement"]
    assert agreement["queries_with_multiple_judges"] == 1
    # relevance: vals=[5,3] → pop std = 1.0, pairwise mad = 2.0
    assert agreement["relevance_std"] == 1.0
    assert agreement["relevance_pairwise_mad"] == 2.0
    # coverage: vals=[5,5] → std 0, mad 0
    assert agreement["coverage_std"] == 0.0
    assert agreement["coverage_pairwise_mad"] == 0.0


def test_aggregate_no_irr_when_only_one_judge():
    rubrics = [_rub("llm:a", (4, 4, 4, 4)), _rub("llm:a", (2, 2, 2, 2), query="q2")]
    out = aggregate_scores(rubrics)
    assert out["agreement"]["queries_with_multiple_judges"] == 0


def test_run_eval_merges_human_forms(tmp_path):
    """run_eval should load and merge human-eval JSON forms with LLM judgments."""
    from research_trail.cli import run_eval

    input_jsonl = tmp_path / "results.jsonl"
    input_jsonl.write_text(json.dumps({"query": "q", "output": {"papers": []}}) + "\n")

    forms_dir = tmp_path / "forms"
    forms_dir.mkdir()
    (forms_dir / "rev1.json").write_text(
        json.dumps(
            {
                "judge_id": "human:rev1",
                "query": "q",
                "system_output": {},
                "score": {
                    "relevance": 3,
                    "coverage": 3,
                    "structural_organization": 3,
                    "insightfulness": 3,
                    "rationale": "ok",
                },
            }
        )
    )

    output = tmp_path / "agg.json"
    rc = run_eval(["--input", str(input_jsonl), "--output", str(output),
                   "--human-forms", str(forms_dir)])
    assert rc == 0
    payload = json.loads(output.read_text())
    judges = {r["judge_id"] for r in payload["rubrics"]}
    assert any(j.startswith("human:") for j in judges)
    assert any(j.startswith("llm:") for j in judges)
    assert payload["summary"]["n"] == 2


def test_scaffold_human_eval_writes_forms(tmp_path):
    from research_trail.cli import scaffold_human_eval

    input_jsonl = tmp_path / "in.jsonl"
    input_jsonl.write_text(
        json.dumps({"query": "q1", "output": {"papers": []}}) + "\n"
        + json.dumps({"query": "q2", "output": {"papers": []}}) + "\n"
    )
    out_dir = tmp_path / "forms"
    rc = scaffold_human_eval(["--input", str(input_jsonl), "--out-dir", str(out_dir)])
    assert rc == 0
    forms = sorted(out_dir.glob("*.json"))
    assert len(forms) == 2
    data = json.loads(forms[0].read_text())
    assert data["query"] == "q1"
    assert "score" in data

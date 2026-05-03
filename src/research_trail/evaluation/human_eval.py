"""Human-evaluation IO helpers.

Reviewers receive a JSON form, fill in scores in place, and the loader
parses it back into a ``Rubric`` for aggregation alongside LLM judgments.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_trail.evaluation.rubric import Rubric


def write_form_template(query: str, system_output: dict, out_path: Path) -> Path:
    template = {
        "judge_id": "human:<netid>",
        "query": query,
        "system_output": system_output,
        "score": {
            "relevance": 0,
            "coverage": 0,
            "structural_organization": 0,
            "insightfulness": 0,
            "rationale": "",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, indent=2))
    return out_path


def load_human_rubric(path: Path) -> Rubric:
    data = json.loads(Path(path).read_text())
    return Rubric.model_validate(
        {"judge_id": data["judge_id"], "query": data["query"], "score": data["score"]}
    )

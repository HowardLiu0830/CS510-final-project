"""Evaluation harness — Howard's responsibility per proposal."""

from research_trail.evaluation.llm_judge import judge
from research_trail.evaluation.metrics import aggregate_scores
from research_trail.evaluation.rubric import Rubric, RubricScore

__all__ = ["Rubric", "RubricScore", "judge", "aggregate_scores"]

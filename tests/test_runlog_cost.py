"""Tests for the LLM callback handler's token/cost accounting.

We feed it a hand-rolled ``LLMResult``-shaped object so the test doesn't need
real LangChain or OpenAI dependencies at import time.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from research_trail.runlog import LLMCallbackHandler, _current_node


def _fake_response(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"):
    """Mimic the LangChain LLMResult shape that on_llm_end receives."""
    msg = SimpleNamespace(content="hello")
    gen = SimpleNamespace(message=msg, text="hello")
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


def test_handler_accumulates_tokens_and_cost(tmp_path):
    h = LLMCallbackHandler(tmp_path)
    rid = uuid4()
    tok = _current_node.set("scope_query")
    try:
        h.on_chat_model_start(
            {"kwargs": {"model": "gpt-4o-mini"}},
            [[SimpleNamespace(type="human", content="q")]],
            run_id=rid,
        )
        h.on_llm_end(_fake_response(1000, 500), run_id=rid)
    finally:
        _current_node.reset(tok)

    totals = h.totals()
    assert totals["prompt_tokens"] == 1000
    assert totals["completion_tokens"] == 500
    assert totals["total_tokens"] == 1500
    assert totals["cost_usd"] > 0  # gpt-4o-mini is in litellm's pricing table
    assert totals["calls"] == 1

    by_node = h.totals_by_node()
    assert "scope_query" in by_node
    assert by_node["scope_query"]["total_tokens"] == 1500


def test_handler_handles_unknown_model(tmp_path):
    h = LLMCallbackHandler(tmp_path)
    rid = uuid4()
    h.on_chat_model_start(
        {"kwargs": {"model": "totally-made-up-model-xyz"}},
        [[SimpleNamespace(type="human", content="q")]],
        run_id=rid,
    )
    h.on_llm_end(_fake_response(100, 50, model="totally-made-up-model-xyz"), run_id=rid)
    totals = h.totals()
    assert totals["total_tokens"] == 150
    assert totals["cost_usd"] == 0.0
    assert totals["cost_unknown_model"] is True


def test_handler_attributes_to_correct_node(tmp_path):
    h = LLMCallbackHandler(tmp_path)
    for node, p, c in [("scope_query", 100, 50), ("synthesize", 2000, 800)]:
        rid = uuid4()
        tok = _current_node.set(node)
        try:
            h.on_chat_model_start(
                {"kwargs": {"model": "gpt-4o-mini"}},
                [[SimpleNamespace(type="human", content="q")]],
                run_id=rid,
            )
            h.on_llm_end(_fake_response(p, c), run_id=rid)
        finally:
            _current_node.reset(tok)

    by_node = h.totals_by_node()
    assert by_node["scope_query"]["total_tokens"] == 150
    assert by_node["synthesize"]["total_tokens"] == 2800
    assert h.totals()["total_tokens"] == 150 + 2800


def test_handler_folds_reasoning_tokens_into_completion(tmp_path):
    """Reasoning models report hidden reasoning_tokens; they must be billed."""
    h = LLMCallbackHandler(tmp_path)
    rid = uuid4()
    msg = SimpleNamespace(content="x")
    gen = SimpleNamespace(message=msg, text="x")
    response = SimpleNamespace(
        generations=[[gen]],
        llm_output={
            "model_name": "gpt-4o-mini",
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "completion_tokens_details": {"reasoning_tokens": 800},
            },
        },
    )
    h.on_chat_model_start(
        {"kwargs": {"model": "gpt-4o-mini"}},
        [[SimpleNamespace(type="human", content="q")]],
        run_id=rid,
    )
    h.on_llm_end(response, run_id=rid)
    totals = h.totals()
    # 100 visible completion + 800 reasoning = 900 billed completion tokens
    assert totals["completion_tokens"] == 900
    assert totals["prompt_tokens"] == 200


def test_handler_writes_token_fields_to_jsonl(tmp_path):
    import json

    h = LLMCallbackHandler(tmp_path)
    rid = uuid4()
    h.on_chat_model_start(
        {"kwargs": {"model": "gpt-4o-mini"}},
        [[SimpleNamespace(type="human", content="q")]],
        run_id=rid,
    )
    h.on_llm_end(_fake_response(10, 5), run_id=rid)
    line = (tmp_path / "messages.jsonl").read_text().strip()
    rec = json.loads(line)
    assert rec["prompt_tokens"] == 10
    assert rec["completion_tokens"] == 5
    assert rec["total_tokens"] == 15
    assert rec["cost_usd"] > 0

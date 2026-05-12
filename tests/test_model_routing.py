"""Guardrails on model routing.

The judge model (settings.openai_model) is reserved — generation calls
must not borrow it, otherwise the judge would be evaluating itself.
"""

from __future__ import annotations

import os

import pytest

from research_trail.config import get_settings
from research_trail.llm.client import override_model


def test_override_model_rejects_judge_model(monkeypatch):
    # Conftest forces offline mode by clearing keys; here we just need
    # settings.openai_model to be readable, which it always is.
    judge = get_settings().openai_model
    with pytest.raises(ValueError, match="reserved for the judge"):
        with override_model(judge):
            pass


def test_override_model_accepts_a_different_model():
    judge = get_settings().openai_model
    other = "definitely-not-the-judge-model"
    assert other != judge
    # Should not raise and should leave no lingering override after the block.
    from research_trail.llm.client import _model_override

    assert _model_override.get() is None
    with override_model(other):
        assert _model_override.get() == other
    assert _model_override.get() is None


def test_override_model_none_is_noop():
    with override_model(None):
        pass  # must not raise

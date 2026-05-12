"""LLM client factory.

Returns a LangChain ChatOpenAI in online mode and ``None`` when offline so
callers can branch to deterministic stubs without any network access.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

from research_trail.config import get_settings

# When set (via override_model), get_chat_model uses this name instead of
# settings.openai_model. Lets a single CLI invocation pin the *generation*
# model while leaving the judge's model untouched.
_model_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "research_trail_model_override", default=None
)


@contextmanager
def override_model(model: str | None):
    """Pin the generation model for the duration of a block.

    ``None`` is a no-op so callers can pass an optional CLI flag through
    unconditionally. Refuses to set the override to the judge model
    (``settings.openai_model``) — that model is reserved so generation
    and evaluation stay strictly independent.
    """
    if model is None:
        yield
        return
    judge_model = get_settings().openai_model
    if model == judge_model:
        raise ValueError(
            f"Model {model!r} is reserved for the judge (OPENAI_MODEL in .env). "
            "Pick a different model for generation, or change OPENAI_MODEL first."
        )
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    settings = get_settings()
    if settings.offline:
        return None

    from langchain_openai import ChatOpenAI

    from research_trail.runlog import get_current_handler

    handler = get_current_handler()
    callbacks = [handler] if handler is not None else None

    chosen = model or _model_override.get() or settings.openai_model

    # Per-model routing: a "/" in the model name (e.g. "deepseek/...", "openai/...")
    # is OpenRouter's namespacing convention, so we route those to the OpenRouter
    # endpoint with the OpenRouter key. Bare OpenAI model names (gpt-4o-mini,
    # gpt-5-mini, ...) go to OpenAI direct. An explicit OPENAI_BASE_URL still
    # wins so the override remains available for unusual setups.
    explicit_base = settings.openai_base_url
    if explicit_base:
        base = explicit_base
        api_key = (
            settings.openrouter_api_key
            if "openrouter" in explicit_base and settings.openrouter_api_key
            else settings.openai_api_key
        )
    elif "/" in chosen and settings.openrouter_api_key:
        base = "https://openrouter.ai/api/v1"
        api_key = settings.openrouter_api_key
    else:
        base = None  # default OpenAI endpoint
        api_key = settings.openai_api_key

    return ChatOpenAI(
        model=chosen,
        temperature=temperature,
        api_key=api_key,
        base_url=base,
        callbacks=callbacks,
    )

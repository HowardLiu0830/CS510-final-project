"""LLM client factory.

Returns a LangChain ChatOpenAI in online mode and ``None`` when offline so
callers can branch to deterministic stubs without any network access.
"""

from __future__ import annotations

from research_trail.config import get_settings


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    settings = get_settings()
    if settings.offline:
        return None

    from langchain_openai import ChatOpenAI

    from research_trail.runlog import get_current_handler

    handler = get_current_handler()
    callbacks = [handler] if handler is not None else None

    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        callbacks=callbacks,
    )

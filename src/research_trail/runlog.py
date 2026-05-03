"""Per-run persistence: state.json, graph.json, messages.jsonl, events.log, meta.json.

A "run" is one invocation of the research-trail graph. Use ``open_run(query)`` as
a context manager around ``graph.invoke``; it installs an LLM callback so every
prompt/response pair lands in messages.jsonl, attaches a file handler so per-node
trace lines land in events.log, and writes meta.json on exit.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from research_trail.config import PROJECT_ROOT, get_settings

_RUNS_ROOT = PROJECT_ROOT / "data" / "runs"
_LOGGER_NAME = "research_trail"
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_current_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "research_trail_current_node", default=None
)
_current_handler: contextvars.ContextVar["LLMCallbackHandler | None"] = contextvars.ContextVar(
    "research_trail_current_handler", default=None
)


def _slugify(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "query"


def make_run_dir(query: str, root: Path | None = None) -> Path:
    """Create and return a fresh ``data/runs/<ts>__<slug>/`` directory."""
    base = root or _RUNS_ROOT
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = base / f"{ts}__{_slugify(query)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_current_handler() -> "LLMCallbackHandler | None":
    """Return the LLM callback handler bound to the active run, if any."""
    return _current_handler.get()


def serialize_state(query: str, state: dict) -> dict:
    """Convert a final ResearchState to a JSON-serializable dict."""
    return {
        "query": query,
        "sub_problems": state.get("sub_problems", []),
        "papers": [_dump(p) for p in state.get("papers", [])],
        "extractions": [_dump(e) for e in state.get("extractions", [])],
        "graph": state.get("graph", {}),
        "summary": state.get("summary", ""),
    }


def write_state(run_dir: Path, serial: dict) -> None:
    """Write state.json (full) and graph.json (just the graph) to ``run_dir``."""
    (run_dir / "state.json").write_text(
        json.dumps(serial, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "graph.json").write_text(
        json.dumps(serial.get("graph", {}), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _dump(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _model_name(serialized: dict | None) -> str | None:
    if not serialized:
        return None
    kw = serialized.get("kwargs") or {}
    return kw.get("model") or kw.get("model_name") or serialized.get("name")


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    parts.extend(str(b) for b in content)
                else:
                    parts.append(str(content))
            else:
                parts.append(getattr(gen, "text", str(gen)))
    return "\n".join(parts) if parts else str(response)


class LLMCallbackHandler(BaseCallbackHandler):
    """Append every LLM round-trip (prompt + completion + latency) to messages.jsonl."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "messages.jsonl"
        self._path.touch(exist_ok=True)
        self._open: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._open[str(run_id)] = {
            "kind": "llm",
            "node": _current_node.get(),
            "model": _model_name(serialized),
            "prompt": "\n\n".join(prompts),
            "t0": time.perf_counter(),
        }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat: list[str] = []
        for batch in messages:
            for m in batch:
                role = getattr(m, "type", m.__class__.__name__)
                content = getattr(m, "content", str(m))
                flat.append(f"[{role}] {content}")
        self._open[str(run_id)] = {
            "kind": "chat",
            "node": _current_node.get(),
            "model": _model_name(serialized),
            "prompt": "\n\n".join(flat),
            "t0": time.perf_counter(),
        }

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        meta = self._open.pop(str(run_id), None)
        if meta is None:
            return
        self._append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "node": meta["node"],
                "kind": meta["kind"],
                "model": meta["model"],
                "prompt": meta["prompt"],
                "response": _extract_text(response),
                "latency_ms": int((time.perf_counter() - meta["t0"]) * 1000),
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        meta = self._open.pop(str(run_id), None)
        if meta is None:
            return
        self._append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "node": meta["node"],
                "kind": meta["kind"],
                "model": meta["model"],
                "prompt": meta["prompt"],
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": int((time.perf_counter() - meta["t0"]) * 1000),
            }
        )

    def _append(self, rec: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _setup_event_logger(run_dir: Path) -> logging.Handler:
    fh = logging.FileHandler(run_dir / "events.log", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)sZ [%(levelname)s] %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    fh.setLevel(logging.INFO)
    log = logging.getLogger(_LOGGER_NAME)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)
    log.addHandler(fh)
    return fh


def _teardown_event_logger(handler: logging.Handler) -> None:
    logging.getLogger(_LOGGER_NAME).removeHandler(handler)
    handler.close()


def _redact(value: str | None) -> str | None:
    return "<redacted>" if value else None


def _package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("research_trail")
    except Exception:
        return None


def _write_meta(run_dir: Path, *, query: str, started_at: float, ended_at: float) -> None:
    s = get_settings()
    meta = {
        "query": query,
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
        "duration_s": round(ended_at - started_at, 3),
        "offline": s.offline,
        "model": s.openai_model,
        "openai_api_key": _redact(s.openai_api_key),
        "s2_api_key": _redact(s.s2_api_key),
        "openalex_email": s.openalex_email,
        "package_version": _package_version(),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


@contextmanager
def open_run(query: str, *, run_dir: Path | None = None):
    """Set up per-run logging + persistence around ``graph.invoke``.

    Yields the run directory. On exit writes meta.json. Caller is responsible for
    calling ``write_state`` (with a serialized state dict) before exiting if
    state.json / graph.json are wanted.
    """
    started = time.time()
    rd = run_dir or make_run_dir(query)
    rd.mkdir(parents=True, exist_ok=True)
    handler = LLMCallbackHandler(rd)
    log_handler = _setup_event_logger(rd)
    h_token = _current_handler.set(handler)
    log = logging.getLogger(_LOGGER_NAME)
    log.info("run start | query=%r dir=%s offline=%s", query, rd, get_settings().offline)
    try:
        yield rd
    finally:
        ended = time.time()
        log.info("run end | duration=%.2fs", ended - started)
        _write_meta(rd, query=query, started_at=started, ended_at=ended)
        _current_handler.reset(h_token)
        _teardown_event_logger(log_handler)


def node(name: str):
    """Tag a graph node so its LLM calls land labelled and start/end land in events.log."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            token = _current_node.set(name)
            log = logging.getLogger(f"{_LOGGER_NAME}.nodes.{name}")
            log.info("start | state keys=%s", sorted(state.keys()))
            try:
                out = fn(state)
                if isinstance(out, dict):
                    summary = {
                        k: (len(v) if isinstance(v, (list, dict)) else type(v).__name__)
                        for k, v in out.items()
                    }
                    log.info("end | produced=%s", summary)
                else:
                    log.info("end | type=%s", type(out).__name__)
                return out
            except Exception as exc:
                log.exception("error: %s", exc)
                raise
            finally:
                _current_node.reset(token)

        return wrapper

    return deco

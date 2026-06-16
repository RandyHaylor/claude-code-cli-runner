"""A small on-disk registry: reusable-context ``chunk_id -> primed_session_id``.

This backs the OPTIONAL prime-once / fork-per-task session-reuse primitive. The
first time a given ``chunk_id`` is seen, the runner primes a claude session with
that chunk and records the resulting (caller-chosen) session id here; subsequent
tasks carrying the same ``chunk_id`` skip priming and fork directly from the
recorded session.

Format: a single JSON object file mapping each chunk_id to a small record
``{"session_id": <psid>, "source_jsonl": <abs path of the primed jsonl>}``. For
back-compat, a bare-string value (an older ``{chunk_id: primed_session_id}``
file) is still read as a session id with no recorded source jsonl. The default
location is ``~/.claude_code_cli_runner/reusable_sessions.json``; the path is
fully overridable (tests point it at a tmp dir).

Concurrency: writes use a same-directory atomic replace under a process-wide
lock, which is sufficient for this best-effort, single-host use. Reuse is never
correctness-critical, so a lost update at worst causes a redundant prime.
"""

from __future__ import annotations

import json
import os
import threading

DEFAULT_REGISTRY_DIR = os.path.join(
    os.path.expanduser("~"), ".claude_code_cli_runner"
)
DEFAULT_REGISTRY_FILENAME = "reusable_sessions.json"

_LOCK = threading.Lock()


def default_registry_path() -> str:
    """The default registry file path (overridable via env or an explicit arg)."""
    override = os.environ.get("CLAUDE_CODE_CLI_RUNNER_REGISTRY_PATH")
    if override:
        return override
    return os.path.join(DEFAULT_REGISTRY_DIR, DEFAULT_REGISTRY_FILENAME)


def _read_all(registry_path: str) -> dict:
    if not os.path.isfile(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_record(value) -> "dict | None":
    """Normalise a stored value to ``{"session_id", "source_jsonl"}`` or None.

    Back-compat: a bare non-empty string is an older session-id-only entry.
    """
    if isinstance(value, str) and value:
        return {"session_id": value, "source_jsonl": None}
    if isinstance(value, dict):
        sid = value.get("session_id")
        if isinstance(sid, str) and sid:
            source = value.get("source_jsonl")
            return {
                "session_id": sid,
                "source_jsonl": source if isinstance(source, str) and source else None,
            }
    return None


def get_primed_session_id(chunk_id: str, registry_path: "str | None" = None) -> "str | None":
    """Return the recorded primed session id for ``chunk_id``, or None."""
    record = get_primed_record(chunk_id, registry_path=registry_path)
    return record["session_id"] if record else None


def get_primed_record(chunk_id: str, registry_path: "str | None" = None) -> "dict | None":
    """Return ``{"session_id", "source_jsonl"}`` for ``chunk_id``, or None."""
    registry_path = registry_path or default_registry_path()
    with _LOCK:
        value = _read_all(registry_path).get(chunk_id)
    return _coerce_record(value)


def _write_all(registry_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    tmp_path = registry_path + ".tmp.%d" % os.getpid()
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, registry_path)


def record_primed_session_id(
    chunk_id: str,
    primed_session_id: str,
    registry_path: "str | None" = None,
    *,
    source_jsonl: "str | None" = None,
) -> None:
    """Record ``chunk_id -> {session_id, source_jsonl}`` via atomic replace."""
    registry_path = registry_path or default_registry_path()
    with _LOCK:
        data = _read_all(registry_path)
        data[chunk_id] = {
            "session_id": primed_session_id,
            "source_jsonl": source_jsonl,
        }
        _write_all(registry_path, data)


def forget_chunk(chunk_id: str, registry_path: "str | None" = None) -> None:
    """Drop ``chunk_id`` from the registry (e.g. when its session is unusable)."""
    registry_path = registry_path or default_registry_path()
    with _LOCK:
        if not os.path.isfile(registry_path):
            return
        data = _read_all(registry_path)
        if chunk_id in data:
            del data[chunk_id]
            tmp_path = registry_path + ".tmp.%d" % os.getpid()
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, registry_path)

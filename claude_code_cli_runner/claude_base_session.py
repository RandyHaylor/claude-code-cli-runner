"""Warm, dated BASE session that every claude task's FIRST run forks from, so the
universal Unharness init prompt (~30k tokens) is a CACHE READ rather than re-paid on
each new session (raw-1211/1212).

CLAUDE-ONLY by design (adapter isolation, raw-1217): opencode and the runner core never
see this — it lives entirely in the claude adapter path.

Lifecycle: a single base session is kept in a small on-disk store with its creation
time. A new base is primed with the universal init prompt and recorded when a base is
needed and any of these hold (raw-1362):
  * none exists, or its transcript went missing;
  * the existing one is older than the max age (default 3 days; override via
    ``UNHARNESS_CLAUDE_BASE_SESSION_MAX_AGE_DAYS``) — this bounds how long a stale
    session template (built against older dependencies / prompting) can linger;
  * Unharness has been IDLE longer than the max-idle window (default 1 hour; override
    via ``UNHARNESS_CLAUDE_BASE_SESSION_MAX_IDLE_SECONDS``) as of this dispatch — after
    a long idle the base's forkable cache has expired anyway, so a fresh base costs
    nothing extra AND picks up any interim updates. The idle duration is MEASURED BY
    THE CALLER (Unharness's supervisor, across all node activity) and passed in; when
    it is unknown (None) the idle gate is simply skipped (raw-1368).
This is BEST-EFFORT: any failure means the caller simply runs the task as a plain new
session (no fork), never blocking work.

Store format: a single JSON object ``{"session_id", "created_epoch", "source_jsonl"}``.
Default path ``~/.claude_code_cli_runner/base_session.json`` (overridable for tests via
``UNHARNESS_CLAUDE_BASE_SESSION_STORE_PATH`` or an explicit arg).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

# Verbatim universal init prompt (raw-1211): the base session is primed with EXACTLY
# this so every forked task session starts from the same confirmed initialization.
UNIVERSAL_INIT_PROMPT = (
    'You are an agent working within the "Unharness" task management system that '
    "delegates roles and tasks to agent harnesses. You will be given explicit "
    "instructions to follow - confirm by responding with ONLY the json output: "
    "{initialization-confirmed:true}"
)

DEFAULT_BASE_SESSION_MAX_AGE_DAYS = 3.0
# After Unharness has been idle this long as of a dispatch, the warm base's forkable
# cache has expired anyway, so the next task regenerates a fresh base (raw-1362). One
# hour, matching the observed ~30–60 min warm-cache window with headroom.
DEFAULT_BASE_SESSION_MAX_IDLE_SECONDS = 60.0 * 60.0
_DEFAULT_STORE_DIR = os.path.join(os.path.expanduser("~"), ".claude_code_cli_runner")
_DEFAULT_STORE_FILENAME = "base_session.json"
_LOCK = threading.Lock()


def default_base_session_store_path() -> str:
    """The base-session store file path (env-overridable, else the default)."""
    override = os.environ.get("UNHARNESS_CLAUDE_BASE_SESSION_STORE_PATH")
    if override:
        return override
    return os.path.join(_DEFAULT_STORE_DIR, _DEFAULT_STORE_FILENAME)


def resolve_base_session_max_age_seconds() -> float:
    """Max base-session age before it is regenerated (raw-1211: default 5 days),
    overridable via ``UNHARNESS_CLAUDE_BASE_SESSION_MAX_AGE_DAYS``. A malformed value
    falls back to the default rather than crashing."""
    raw = os.environ.get("UNHARNESS_CLAUDE_BASE_SESSION_MAX_AGE_DAYS")
    days = DEFAULT_BASE_SESSION_MAX_AGE_DAYS
    if raw:
        try:
            days = float(raw)
        except (TypeError, ValueError):
            days = DEFAULT_BASE_SESSION_MAX_AGE_DAYS
    return days * 24.0 * 60.0 * 60.0


def resolve_base_session_max_idle_seconds() -> float:
    """Max Unharness idle time (seconds) before the base is regenerated on the next
    dispatch (raw-1362: default 1 hour), overridable via
    ``UNHARNESS_CLAUDE_BASE_SESSION_MAX_IDLE_SECONDS``. A malformed value falls back to
    the default rather than crashing."""
    raw = os.environ.get("UNHARNESS_CLAUDE_BASE_SESSION_MAX_IDLE_SECONDS")
    seconds = DEFAULT_BASE_SESSION_MAX_IDLE_SECONDS
    if raw:
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            seconds = DEFAULT_BASE_SESSION_MAX_IDLE_SECONDS
    return seconds


def _read_store(store_path: str) -> "dict | None":
    if not os.path.isfile(store_path):
        return None
    try:
        with open(store_path, "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    session_id = parsed.get("session_id")
    created_epoch = parsed.get("created_epoch")
    source_jsonl = parsed.get("source_jsonl")
    if not (isinstance(session_id, str) and session_id):
        return None
    if not isinstance(created_epoch, (int, float)):
        return None
    return {
        "session_id": session_id,
        "created_epoch": float(created_epoch),
        "source_jsonl": source_jsonl if isinstance(source_jsonl, str) and source_jsonl else None,
    }


def _write_store(store_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    tmp_path = store_path + ".tmp.%d" % os.getpid()
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, store_path)


def _base_session_is_fresh(
    record: "dict | None",
    *,
    now_epoch: float,
    unharness_idle_seconds: "float | None" = None,
) -> bool:
    """True when ``record`` exists, has a locatable source jsonl, is younger than the
    configured max age, AND Unharness has not been idle past the max-idle window.

    ``unharness_idle_seconds`` is how long Unharness had been idle (no node activity)
    as of this dispatch, measured by the caller; None means unknown and skips the idle
    gate (raw-1362/1368)."""
    if not record or not record.get("source_jsonl"):
        return False
    if not os.path.isfile(record["source_jsonl"]):
        return False
    if (now_epoch - record["created_epoch"]) >= resolve_base_session_max_age_seconds():
        return False
    if (
        unharness_idle_seconds is not None
        and unharness_idle_seconds >= resolve_base_session_max_idle_seconds()
    ):
        return False
    return True


def ensure_fresh_base_session(
    *,
    prime_base_session,
    session_jsonl_path_for,
    store_path: "str | None" = None,
    now_epoch: "float | None" = None,
    unharness_idle_seconds: "float | None" = None,
) -> dict:
    """Return the current warm base session record, priming a new one first if none
    exists or the existing one is stale/missing/idle-expired.

    ``unharness_idle_seconds`` (optional): how long Unharness had been idle across all
    node activity as of this dispatch; when it exceeds the max-idle window the base is
    regenerated even if still within its max age (raw-1362). None => idle gate skipped.

    Injected callables keep this pure/testable (no claude dependency here):
      * ``prime_base_session(base_session_id) -> None`` — run a self-completing claude
        that ingests UNIVERSAL_INIT_PROMPT under ``base_session_id``, leaving a forkable
        transcript. Raises on failure.
      * ``session_jsonl_path_for(base_session_id) -> str`` — where that transcript lands.

    Returns ``{"session_id", "created_epoch", "source_jsonl"}``. Raises on prime
    failure (the caller falls back to a plain, unforked run)."""
    store_path = store_path or default_base_session_store_path()
    now_epoch = time.time() if now_epoch is None else now_epoch
    with _LOCK:
        record = _read_store(store_path)
        if _base_session_is_fresh(
            record,
            now_epoch=now_epoch,
            unharness_idle_seconds=unharness_idle_seconds,
        ):
            return record
        # Missing or stale -> prime a fresh base with the universal init prompt.
        base_session_id = str(uuid.uuid4())
        prime_base_session(base_session_id)
        record = {
            "session_id": base_session_id,
            "created_epoch": now_epoch,
            "source_jsonl": session_jsonl_path_for(base_session_id),
        }
        _write_store(store_path, record)
        return record


def forget_base_session(store_path: "str | None" = None) -> None:
    """Drop the recorded base session (e.g. when it proved unforkable), so the next
    call primes a new one."""
    store_path = store_path or default_base_session_store_path()
    with _LOCK:
        try:
            if os.path.isfile(store_path):
                os.remove(store_path)
        except OSError:
            pass

"""Keep-alive registry (raw-830): the fail-safe against orphaned harness runs.

Unharness sends a tiny keep-alive signal for every task that is currently
IN PROGRESS (and ONLY for in-progress tasks) on every 5th engine heartbeat
(~every 10 seconds). The serve records each signal here by task id. A run
launched with ``keep_alive_expected`` checks this registry and AGGRESSIVELY
kills its harness process group when no signal has arrived within the
timeout (default 60 seconds) — so a run whose orchestrator died, whose task
was wiped, or whose task left in_progress can never keep burning the GPU.
"""

from __future__ import annotations

import threading
import time

_registry_lock = threading.Lock()
_last_keep_alive_monotonic_by_task_id: "dict[str, float]" = {}


def record_keep_alive_signal(task_id: str) -> None:
    """Record that a keep-alive signal arrived for ``task_id`` just now."""
    with _registry_lock:
        _last_keep_alive_monotonic_by_task_id[task_id] = time.monotonic()


def seconds_since_last_keep_alive(task_id: str) -> "float | None":
    """Seconds since the last keep-alive for ``task_id``; None if never seen."""
    with _registry_lock:
        last_signal = _last_keep_alive_monotonic_by_task_id.get(task_id)
    if last_signal is None:
        return None
    return time.monotonic() - last_signal


def clear_keep_alive_record(task_id: str) -> None:
    """Drop the record when a run ends (bookkeeping hygiene)."""
    with _registry_lock:
        _last_keep_alive_monotonic_by_task_id.pop(task_id, None)

"""The on-disk live-window contract: file names, run-state values, control
intents, and the path/IO helpers around them.

These names and intents are LOAD-BEARING: external readers (dashboards, tailers)
agree with this runner on exactly these on-disk file names and control intents.
They are kept here, in one place, so every face of the tool shares them.

  - ``task_live_log.jsonl``       append-only JSONL; one record per stream chunk
  - ``task_control_channel.jsonl``append-only JSONL an external reader WRITES
                                  control intents into
  - ``task_run_status.json``      a tiny sidecar reflecting the out-of-band run
                                  state (running / paused / operator_ended)

Control intents: pause | resume | send_command | end_and_return.
"""

from __future__ import annotations

import json
import os
import time

# Role-named per-task files under the workspace. EXACT names are part of the
# public live-window contract — do not rename.
LIVE_LOG_FILE_NAME = "task_live_log.jsonl"
CONTROL_CHANNEL_FILE_NAME = "task_control_channel.jsonl"
RUN_STATUS_FILE_NAME = "task_run_status.json"

# Out-of-band run-state annotation values (descriptions of a live process for a
# UI, never queue/runner states).
RUN_STATE_RUNNING = "running"
RUN_STATE_PAUSED = "paused"
RUN_STATE_OPERATOR_ENDED = "operator_ended"

# Control intents an external reader may write into the control channel.
CONTROL_PAUSE = "pause"
CONTROL_RESUME = "resume"
CONTROL_SEND_COMMAND = "send_command"
CONTROL_END_AND_RETURN = "end_and_return"

KNOWN_CONTROL_INTENTS = (
    CONTROL_PAUSE,
    CONTROL_RESUME,
    CONTROL_SEND_COMMAND,
    CONTROL_END_AND_RETURN,
)


def live_log_path(workspace_directory: str) -> str:
    """Absolute path of the per-task live log file under the workspace."""
    return os.path.join(workspace_directory, LIVE_LOG_FILE_NAME)


def control_channel_path(workspace_directory: str) -> str:
    """Absolute path of the per-task control channel file under the workspace."""
    return os.path.join(workspace_directory, CONTROL_CHANNEL_FILE_NAME)


def run_status_path(workspace_directory: str) -> str:
    """Absolute path of the per-task run-status sidecar under the workspace."""
    return os.path.join(workspace_directory, RUN_STATUS_FILE_NAME)


def reflect_run_state(workspace_directory: str, run_state: str) -> str:
    """Write the out-of-band run-state annotation to the workspace sidecar.

    Returns the path written. Purely an annotation about the running process.
    """
    path = run_status_path(workspace_directory)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"run_state": run_state, "updated_at": time.time()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_new_control_intents(
    control_path: str, already_consumed: int
) -> "tuple[list, int]":
    """Return (new control-intent dicts, new consumed-line count).

    The control channel is append-only JSONL an external reader writes to. We
    read every line, skip the ones already consumed, and parse the rest.
    Unparseable or non-dict lines are ignored (defensive, never fabricated).
    """
    if not os.path.isfile(control_path):
        return [], already_consumed
    with open(control_path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    new_intents = []
    for line in lines[already_consumed:]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "control_intent" in parsed:
            new_intents.append(parsed)
    return new_intents, len(lines)


def build_control_intent(control: str, command_text: "str | None" = None) -> dict:
    """Validate and build a control-intent record for the control channel.

    Carries ``{"control_intent": <intent>}`` plus, for ``send_command``, a
    ``command_text`` string. Raises ValueError for an unknown intent or a
    send_command missing its text.
    """
    if control not in KNOWN_CONTROL_INTENTS:
        raise ValueError(
            "unknown control intent %r; expected one of %s"
            % (control, ", ".join(KNOWN_CONTROL_INTENTS))
        )
    intent: dict = {"control_intent": control}
    if control == CONTROL_SEND_COMMAND:
        if not isinstance(command_text, str) or not command_text.strip():
            raise ValueError(
                "a 'send_command' intent requires a non-empty 'command_text'"
            )
        intent["command_text"] = command_text
    return intent


def append_control_intent(
    workspace_directory: str, control: str, command_text: "str | None" = None
) -> dict:
    """Append ONE control intent to the task's control channel as a flushed
    JSONL line. Creates the workspace directory if absent. Returns the intent
    written. This is the entire out-of-band control mechanism for writers."""
    intent = build_control_intent(control, command_text)
    os.makedirs(workspace_directory, exist_ok=True)
    path = control_channel_path(workspace_directory)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(intent) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return intent

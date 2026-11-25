"""The generic, harness-agnostic request "continue this task in a fresh session from a
handoff document" (docs/SESSION-HANDOFF-INTERFACE.md).

The agent sends it as a fenced block in its turn text (never a harness tool call, so no
harness hook can block it):

    ```runner-request
    {"request": "continue_this_task_in_a_fresh_session_from_a_handoff_document",
     "handoff_document_path": "/abs/path.md"}
    ```

This module holds the request shape, its detection and validation, the messages the
runner sends in response, and the Claude attention-hook settings fragment. The runner core
(runner.py) acts on an accepted request through the harness integration's
``start_fresh_session_from_handoff_document``.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import json

from .context_threshold_handoff_attention_hook import (
    CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST,
    GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE,
    RUNNER_REQUEST_FENCE_LANGUAGE,
    resolve_gate_script_path,
)
from .request import LOCATION_LOCAL_SUBPROCESS

__all__ = [
    "CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST",
    "GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE",
    "HANDOFF_DOCUMENT_DIRECTORY_NAME",
]

HANDOFF_DOCUMENT_DIRECTORY_NAME = "session_handoff_documents"
ATTENTION_HOOK_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "context_threshold_handoff_attention_hook.py"
)

_RUNNER_REQUEST_FENCED_BLOCK_PATTERN = re.compile(
    r"```[ \t]*" + re.escape(RUNNER_REQUEST_FENCE_LANGUAGE) + r"[ \t]*\n(.*?)```",
    re.DOTALL,
)


def detect_fresh_session_handoff_requests_in_turn_text(turn_text: str) -> "list[dict]":
    """Every well-formed runner-request block (a JSON object with a string ``request``)
    in one turn's text, in order. Malformed blocks are ignored."""
    detected = []
    for body in _RUNNER_REQUEST_FENCED_BLOCK_PATTERN.findall(turn_text or ""):
        try:
            candidate = json.loads(body.strip())
        except ValueError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("request"), str):
            detected.append(candidate)
    return detected


def resolve_handoff_document_path(request: dict, working_directory: str) -> str:
    path = str(request.get("handoff_document_path") or "")
    if path and not os.path.isabs(path):
        path = os.path.join(working_directory, path)
    return path


def reason_the_handoff_request_is_refused(request: dict, working_directory: str) -> "str | None":
    """None when the request can be carried out, else the refusal reason sent back."""
    if request.get("request") != CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST:
        return "unknown request %r; the runner accepts %r" % (
            request.get("request"), CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST)
    path = resolve_handoff_document_path(request, working_directory)
    if not path:
        return "the request has no handoff_document_path"
    if not os.path.isfile(path):
        return "the handoff document %s does not exist; write it first, then ask again" % path
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        if not handle.read().strip():
            return "the handoff document %s is empty; write it first, then ask again" % path
    return None


def compose_refusal_message_text(reason: str) -> str:
    return "runner-request refused: %s" % reason


def compose_fresh_session_first_message_text(handoff_document_path: str) -> str:
    return (
        "You are continuing a task that an earlier session started; that session ran out "
        "of context and handed off to you. Read the handoff document at %s first, then "
        "continue the work from where it stops. Finish the task exactly as the original "
        "task asked, including its reply format." % handoff_document_path
    )


def reason_the_attention_hook_cannot_be_installed(run_request) -> "str | None":
    """None when the context-threshold attention hook can be installed for this run."""
    if not run_request.fresh_session_handoff_context_token_threshold:
        return "no fresh_session_handoff_context_token_threshold set for this run"
    if run_request.execution_location != LOCATION_LOCAL_SUBPROCESS:
        return (
            "the attention hook runs only where the runner runs (local_subprocess); "
            "this run is %s" % run_request.execution_location
        )
    gate_path = resolve_gate_script_path()
    if not os.path.isfile(gate_path):
        return "the claude-code-token-count gate script was not found at %s" % gate_path
    return None


def add_attention_hook_to_claude_settings(run_request, claude_settings: dict) -> dict:
    """Return ``claude_settings`` with the PreToolUse attention hook appended (the
    caller's own hooks are kept). Unchanged when the hook cannot be installed."""
    if reason_the_attention_hook_cannot_be_installed(run_request):
        return claude_settings
    handoff_document_directory = os.path.join(
        os.fspath(run_request.workspace_directory), HANDOFF_DOCUMENT_DIRECTORY_NAME
    )
    command = " ".join(shlex.quote(part) for part in (
        sys.executable, ATTENTION_HOOK_SCRIPT_PATH,
        "--threshold", str(int(run_request.fresh_session_handoff_context_token_threshold)),
        "--handoff-document-directory", handoff_document_directory,
    ))
    merged = dict(claude_settings)
    hooks = dict(merged.get("hooks") or {})
    pre_tool_use = list(hooks.get("PreToolUse") or [])
    pre_tool_use.append({"matcher": "", "hooks": [{"type": "command", "command": command}]})
    hooks["PreToolUse"] = pre_tool_use
    merged["hooks"] = hooks
    return merged

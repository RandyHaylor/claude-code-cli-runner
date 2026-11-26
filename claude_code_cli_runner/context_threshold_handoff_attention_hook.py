#!/usr/bin/env python3
"""Claude Code PreToolUse hook: past a per-run context threshold, refuse tool calls with a
message telling the agent the handoff procedure (docs/SESSION-HANDOFF-INTERFACE.md).

The run is not blocked: the refusal only gets the agent's attention. The one tool call let
through past the threshold is a Write of exactly this session's handoff document. The
agent then sends the runner a text request (not a tool call, so this hook never sees it)
to continue the task in a fresh session from that document.

The context size is measured by the claude-code-token-count gate's own ``read_last_usage``
(last ``message.usage`` in the transcript, ``all_observed_tokens``), imported from the gate
script. Standalone on purpose: it runs as a script and imports nothing from the runner.

Usage (installed by the runner through its inline --settings):
    python3 context_threshold_handoff_attention_hook.py --threshold N \
        --handoff-document-directory DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE = "CLAUDE_CODE_TOKEN_COUNT_GATE_SCRIPT"
DEFAULT_GATE_SCRIPT_PATH = (
    "~/.claude/skills/claude-code-token-count/scripts/token_usage_gate.py"
)
RUNNER_REQUEST_FENCE_LANGUAGE = "runner-request"
CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST = (
    "continue_this_task_in_a_fresh_session_from_a_handoff_document"
)
HANDOFF_DOCUMENT_WRITING_TOOL_NAME = "Write"


def resolve_gate_script_path() -> str:
    return os.path.expanduser(
        os.environ.get(GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE) or DEFAULT_GATE_SCRIPT_PATH
    )


def handoff_document_path_for_session(handoff_document_directory: str, session_id: str) -> str:
    safe_session_id = "".join(
        character for character in (session_id or "unknown") if character.isalnum() or character in "-_"
    ) or "unknown"
    return os.path.join(
        handoff_document_directory, "handoff_from_session_%s.md" % safe_session_id
    )


def compose_handoff_procedure_text(tokens: int, threshold: int, handoff_document_path: str) -> str:
    request_body = json.dumps({
        "request": CONTINUE_IN_FRESH_SESSION_FROM_HANDOFF_DOCUMENT_REQUEST,
        "handoff_document_path": handoff_document_path,
    })
    return (
        "CONTEXT THRESHOLD REACHED (%d of %d tokens). This tool call was refused to get "
        "your attention; the task is NOT over. Follow the handoff procedure now:\n"
        "1. Write a clear handoff document with the Write tool to exactly this path: %s\n"
        "   It must let a new session continue with no other context: the goal, what is "
        "done, what is left, decisions made, files touched, how to verify, open problems.\n"
        "2. Then end your reply with this block and stop (it is a request to the runner, "
        "not a tool call):\n```%s\n%s\n```\n"
        "The runner starts a fresh session for this same task and gives it the document. "
        "Every other tool call is refused until then."
        % (tokens, threshold, handoff_document_path, RUNNER_REQUEST_FENCE_LANGUAGE, request_body)
    )


def _load_read_last_usage():
    gate_path = resolve_gate_script_path()
    specification = importlib.util.spec_from_file_location("token_usage_gate", gate_path)
    if specification is None or specification.loader is None:
        raise ImportError("gate script not loadable: %s" % gate_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.read_last_usage


def read_first_turn_context_tokens(transcript_path: str) -> int:
    """The context size of the session's FIRST recorded turn (same measure as the gate:
    input + cache creation + cache read + output). A fresh session starts with its system
    prompt and brief already in context; if that alone is at or over the threshold, a
    handoff cannot bring the next session under it.

    Seen live 2026-11-25: with a threshold below the starting context, every fresh session
    was refused on its first tool call and handed off again — an endless handoff loop."""
    try:
        with open(transcript_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                usage = ((entry.get("message") or {}).get("usage")) if isinstance(entry, dict) else None
                if isinstance(usage, dict):
                    return sum(int(usage.get(key) or 0) for key in (
                        "input_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens", "output_tokens"))
    except OSError:
        return 0
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--handoff-document-directory", required=True)
    arguments = parser.parse_args(argv)
    payload = json.load(sys.stdin)

    transcript_path = payload.get("transcript_path") or ""
    tokens = 0
    if transcript_path:
        tokens = int(_load_read_last_usage()(str(transcript_path)).get("all_observed_tokens", 0) or 0)
    if tokens < arguments.threshold:
        return 0  # allow: no output
    if transcript_path and read_first_turn_context_tokens(str(transcript_path)) >= arguments.threshold:
        # The session STARTED over the threshold: a fresh session would too, so asking for
        # a handoff can only loop. Let the work continue and say so once per call.
        print(json.dumps({
            "systemMessage": "Context threshold is below this session's starting context; "
                             "handoff skipped (it would loop).",
        }))
        return 0

    handoff_document_path = handoff_document_path_for_session(
        arguments.handoff_document_directory, payload.get("session_id") or ""
    )
    tool_input = payload.get("tool_input") or {}
    if (
        payload.get("tool_name") == HANDOFF_DOCUMENT_WRITING_TOOL_NAME
        and os.path.abspath(str(tool_input.get("file_path") or "")) == os.path.abspath(handoff_document_path)
    ):
        os.makedirs(os.path.dirname(handoff_document_path), exist_ok=True)
        return 0  # the handoff document itself is let through

    reason = compose_handoff_procedure_text(tokens, arguments.threshold, handoff_document_path)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "systemMessage": "Context threshold reached - handoff procedure requested",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

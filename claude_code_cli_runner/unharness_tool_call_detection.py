"""Detect runner-mediated tool calls in an agent's turn text, and compose the
messages around them (the digestible surface of the runner's MCP integration).

The agent has NO native tool-calling for runner-provided MCP tools — it is PROMPTED
with a known, fixed shape and EMITS a call as ordinary text. The runner recognizes
that shape at the turn's stop, executes the call through the MCP layer, and feeds the
result back as the next turn's input (SoT nd-472/nd-473/nd-474).

The digestible shape is a fenced block the model can emit reliably:

    ```unharness-tool
    {"tool": "read_one_node", "arguments": {"container_id": "nd-9"}}
    ```

- ``tool`` is a DIGESTIBLE tool name (translated to the standard MCP tool name by the
  runner's translation reference before forwarding — nd-473).
- ``arguments`` is the JSON object of arguments (optional; defaults to {}).

Everything outside the fenced block is ordinary turn text and flows through
unchanged (history + the stream up to Unharness).
"""

from __future__ import annotations

import json
import re

UNHARNESS_TOOL_CALL_FENCE_LANGUAGE = "unharness-tool"

# A fenced block: ```unharness-tool\n <json object> \n``` (whitespace-tolerant).
_TOOL_CALL_FENCED_BLOCK_PATTERN = re.compile(
    r"```[ \t]*" + re.escape(UNHARNESS_TOOL_CALL_FENCE_LANGUAGE) + r"[ \t]*\n(.*?)```",
    re.DOTALL,
)


def detect_unharness_tool_calls_in_turn_text(turn_text: str) -> "list[dict]":
    """Find every well-formed unharness tool call in one turn's text, in order.

    Returns a list of ``{"tool": <digestible name>, "arguments": {...}}`` dicts.
    A fenced block whose body is not a JSON object with a string ``tool`` field is
    IGNORED (malformed emissions must not crash the run — the agent simply gets no
    tool result and the turn ends normally)."""
    detected_tool_calls = []
    for fenced_body in _TOOL_CALL_FENCED_BLOCK_PATTERN.findall(turn_text or ""):
        try:
            candidate = json.loads(fenced_body.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(candidate, dict):
            continue
        tool_name = candidate.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        arguments = candidate.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            continue
        detected_tool_calls.append({"tool": tool_name.strip(), "arguments": arguments})
    return detected_tool_calls


def compose_tool_result_turn_message_text(
    digestible_tool_name: str, result_text: str, is_error: bool
) -> str:
    """The text fed back to the agent as its next turn's input after the runner
    executed one tool call — ALWAYS a message, even for empty output or an error,
    so the agent keeps going (raw-1123)."""
    outcome_label = "error" if is_error else "response"
    return "unharness-api command submitted (%s), %s: %s" % (
        digestible_tool_name,
        outcome_label,
        result_text if (result_text or "").strip() else "(no output)",
    )


def compose_unharness_tool_usage_instructions(
    digestible_tool_names: "list[str]",
) -> str:
    """The prompt text that TEACHES an agent the emission shape. Exposed so the
    caller (Unharness) can inject it into an API-only agent's brief; generated from
    the actual tool list so prompt and vocabulary cannot drift."""
    return (
        "UNHARNESS TOOLS: you have no local tools; you operate Unharness by emitting "
        "a tool call as a fenced block in your reply, then STOPPING to wait for the "
        "result, which arrives as your next input. Emit exactly:\n"
        "```" + UNHARNESS_TOOL_CALL_FENCE_LANGUAGE + "\n"
        '{"tool": "<tool name>", "arguments": { ... }}\n'
        "```\n"
        "One call per reply. Available tools: "
        + ", ".join(digestible_tool_names)
        + ". When you are fully done, reply WITHOUT any tool block."
    )

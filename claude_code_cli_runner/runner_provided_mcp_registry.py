"""Registry of the runner-provided MCP tool names (harness-agnostic).

The per-session ``assigned_tool_list`` (see RunRequest) is a GENERIC list that mixes
harness built-in tool names (Read/Bash/Edit/…) with runner-provided MCP names. The
runner is the layer OVER the harnesses and acts as a generic MCP CLIENT: when it sees
one of THESE names in a session's assigned tool list, it knows that entry is a
runner-provided MCP to enable via its MCP-client layer — NOT a harness built-in — so
it holds the name out of the harness built-in tool flag (claude ``--tools``).

Only the NAMES live here; the actual MCP-client mediation (spawning/serving the MCP
and routing tool calls) is a later build step. Reserving the name now lets the tool
split be correct before that layer exists. Add a name here the moment its runner-MCP
becomes available.
"""

from __future__ import annotations

# The Unharness API, exposed to agents as a runner-provided MCP (the runner mediates
# it as the sole executor; harness-agnostic). Reserved now; enablement lands with the
# MCP-client layer.
UNHARNESS_API_RUNNER_MCP_NAME = "unharness-api"

KNOWN_RUNNER_PROVIDED_MCP_NAMES = frozenset({UNHARNESS_API_RUNNER_MCP_NAME})


def split_assigned_tools_into_builtins_and_runner_mcps(
    assigned_tool_list: "list[str]",
) -> "tuple[list[str], list[str]]":
    """Partition a session's generic assigned tool list into ``(builtin_tool_names,
    runner_provided_mcp_names)``, preserving order. A name is a runner-provided MCP
    iff it is in ``KNOWN_RUNNER_PROVIDED_MCP_NAMES``; everything else is treated as a
    harness built-in tool name (passed to the harness's own tool flag)."""
    builtin_tool_names = []
    runner_provided_mcp_names = []
    for tool_name in assigned_tool_list:
        if tool_name in KNOWN_RUNNER_PROVIDED_MCP_NAMES:
            runner_provided_mcp_names.append(tool_name)
        else:
            builtin_tool_names.append(tool_name)
    return builtin_tool_names, runner_provided_mcp_names


# ---- digestible -> standard-MCP tool-name translation (nd-473) ---------------------

_DIGESTIBLE_REFERENCE_FILENAME = "digestible_to_mcp_tool_name_reference.json"


class DigestibleToolNameUnknown(Exception):
    """A detected tool call named a digestible tool with no MCP translation for the
    given runner-provided MCP. Surfaced back to the agent as a tool-result error,
    never a crash."""


def _load_digestible_tool_name_maps() -> dict:
    import json
    import os

    reference_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _DIGESTIBLE_REFERENCE_FILENAME
    )
    with open(reference_path, "r", encoding="utf-8") as reference_file:
        reference = json.load(reference_file)
    return reference.get("mcp_tool_name_by_digestible_name", {})


def list_digestible_tool_names(runner_provided_mcp_name: str) -> "list[str]":
    """The digestible (agent-facing) tool names offered for one runner-provided MCP —
    the vocabulary the agent's prompt teaches. Order follows the reference file."""
    return [
        name
        for name in (_load_digestible_tool_name_maps().get(runner_provided_mcp_name) or {})
        if not name.startswith("_")
    ]


def translate_digestible_tool_name_to_mcp(
    runner_provided_mcp_name: str, digestible_tool_name: str
) -> str:
    """Translate one digestible tool name to the standard MCP tool name for the given
    runner-provided MCP. Raises DigestibleToolNameUnknown for an unmapped name (the
    caller reports it to the agent as a tool error)."""
    this_mcp_map = _load_digestible_tool_name_maps().get(runner_provided_mcp_name) or {}
    mcp_tool_name = this_mcp_map.get(digestible_tool_name)
    if not mcp_tool_name:
        raise DigestibleToolNameUnknown(
            "no %r tool named %r; available: %s"
            % (
                runner_provided_mcp_name,
                digestible_tool_name,
                ", ".join(sorted(k for k in this_mcp_map if not k.startswith("_"))),
            )
        )
    return mcp_tool_name

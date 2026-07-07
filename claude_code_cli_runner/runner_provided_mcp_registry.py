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

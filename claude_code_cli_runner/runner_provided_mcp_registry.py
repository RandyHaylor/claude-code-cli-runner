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
# it as the sole executor; harness-agnostic).
UNHARNESS_API_RUNNER_MCP_NAME = "unharness-api"
# The INGEST-SESSION role-scoped surface of the same MCP server (raw-1353): the
# subset an ingest session needs — reads, the node-building primitives, release,
# abandon — and nothing that can mint a fresh top-level request or requires the
# user-gated approval channel.
UNHARNESS_API_INGEST_RUNNER_MCP_NAME = "unharness-api-ingest"

KNOWN_RUNNER_PROVIDED_MCP_NAMES = frozenset(
    {UNHARNESS_API_RUNNER_MCP_NAME, UNHARNESS_API_INGEST_RUNNER_MCP_NAME}
)


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


def _resolve_unharness_mcp_server_source_directory() -> "str | None":
    """The Unharness MCP server SOURCE checkout this runner can import tool shapes
    from: the ``UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY`` env var when it points at a
    real directory (the run environment, e.g. the VM), else the conventional
    sibling checkout next to this runner's repository (the same relative layout on
    the host and in the VM: ``<repos>/claude-code-cli-runner`` and
    ``<repos>/unharness-mcp-server``). None when neither exists."""
    import os

    candidate = os.environ.get("UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY")
    if candidate and os.path.isdir(os.path.join(candidate, "unharness_mcp_server")):
        return candidate
    runner_repo_directory = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    sibling = os.path.join(
        os.path.dirname(runner_repo_directory), "unharness-mcp-server"
    )
    if os.path.isdir(os.path.join(sibling, "unharness_mcp_server")):
        return sibling
    return None


def describe_digestible_tool_shapes(
    runner_provided_mcp_name: str,
) -> "list[dict] | None":
    """The agent-facing SHAPES (name + description + parameters) of one
    runner-provided MCP's digestible tools, in the reference-file order:

    ``[{"tool_name": <digestible name>, "description": str,
        "parameters": [{"name", "type", "required"}, ...]}, ...]``

    Shapes come from the MCP server package's own introspection (its single
    source of truth — the registered tool functions), mapped MCP-name ->
    digestible-name through the reference file. Returns None when the MCP server
    source is not importable here (the caller falls back to names-only)."""
    import sys

    this_mcp_map = _load_digestible_tool_name_maps().get(runner_provided_mcp_name) or {}
    if not this_mcp_map:
        return None
    source_directory = _resolve_unharness_mcp_server_source_directory()
    if not source_directory:
        return None
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    try:
        from unharness_mcp_server.tool_shape_introspection import (
            describe_tool_shapes_by_name,
        )

        mcp_shapes_by_name = describe_tool_shapes_by_name()
    except Exception:
        return None
    shapes = []
    for digestible_name, mcp_name in this_mcp_map.items():
        if digestible_name.startswith("_"):
            continue
        mcp_shape = mcp_shapes_by_name.get(mcp_name)
        if not mcp_shape:
            continue
        shapes.append(
            {
                "tool_name": digestible_name,
                "description": mcp_shape["description"],
                "parameters": mcp_shape["parameters"],
            }
        )
    return shapes or None


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

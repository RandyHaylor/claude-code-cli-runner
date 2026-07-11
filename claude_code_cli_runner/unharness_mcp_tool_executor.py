"""Execute one translated MCP tool call against the runner-provided Unharness MCP
server — the runner acting as a generic MCP CLIENT (SoT nd-459/nd-460).

The MCP server is STATELESS (nd-470), so the executor spawns it as a stdio subprocess
per call, performs the MCP handshake, issues one ``tools/call``, and returns the
result text. Connection/launch facts come from the run's environment variables (the
existing configuration-not-installation mechanism):

  UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY  directory of the unharness-mcp-server repo
                                         (put on PYTHONPATH; the server runs as
                                         ``python3 -m unharness_mcp_server.mcp_stdio_server_entrypoint``)
  UNHARNESS_AGENT_API_BASE_URL           Unharness endpoint the server loops back to
                                         (role may be baked in, e.g. ".../?role=admin")
  UNHARNESS_AGENT_API_TOKEN              the server's own credential

Failures (missing config, server crash, tool error) come back as ``(text, True)`` —
a tool ERROR RESULT the agent can read and self-correct on, never a runner crash.
"""

from __future__ import annotations

import os
import sys


def _render_expected_signature(tool_shape: dict) -> str:
    """``name(param: type, optional?: type)`` — the exact keys the tool accepts."""
    rendered = ", ".join(
        "%s%s: %s"
        % (p["name"], "" if p.get("required") else "?", p.get("type") or "any")
        for p in (tool_shape.get("parameters") or [])
    )
    return "%s(%s)" % (tool_shape["tool_name"], rendered)


def _validate_arguments_against_shape_or_error(
    mcp_tool_name: str, tool_arguments: dict
) -> "str | None":
    """Check the call's argument KEYS against the tool's introspected shape BEFORE
    spawning the server: an unknown key or a missing required key returns an error
    message that ECHOES the expected signature, so a wrong call self-corrects in
    one round instead of looping on guessed shapes (raw-1351/1353). Returns None
    when the call is well-formed OR when shapes are unavailable here (the server
    then remains the validator — never block on missing introspection)."""
    try:
        from claude_code_cli_runner.runner_provided_mcp_registry import (
            _resolve_unharness_mcp_server_source_directory,
        )

        source_directory = _resolve_unharness_mcp_server_source_directory()
        if not source_directory:
            return None
        if source_directory not in sys.path:
            sys.path.insert(0, source_directory)
        from unharness_mcp_server.tool_shape_introspection import (
            describe_tool_shapes_by_name,
        )

        shape = describe_tool_shapes_by_name().get(mcp_tool_name)
    except Exception:
        return None
    if not shape:
        return None
    allowed_keys = {p["name"] for p in shape["parameters"]}
    required_keys = {p["name"] for p in shape["parameters"] if p["required"]}
    sent_keys = set(tool_arguments or {})
    unknown_keys = sorted(sent_keys - allowed_keys)
    missing_keys = sorted(required_keys - sent_keys)
    if not unknown_keys and not missing_keys:
        return None
    problems = []
    if unknown_keys:
        problems.append("unknown argument key(s): " + ", ".join(unknown_keys))
    if missing_keys:
        problems.append("missing required argument key(s): " + ", ".join(missing_keys))
    return (
        "invalid arguments for %r — %s. The exact signature is: %s. "
        "Resend the call with exactly these keys."
        % (mcp_tool_name, "; ".join(problems), _render_expected_signature(shape))
    )


def execute_unharness_mcp_tool_call(
    mcp_tool_name: str,
    tool_arguments: dict,
    run_environment_variables: dict,
) -> "tuple[str, bool]":
    """Spawn the Unharness MCP server over stdio and execute ONE tools/call.
    Returns ``(result_text, is_error)``. A call whose argument keys do not match
    the tool's introspected shape is rejected up front with the expected
    signature echoed back (no server spawn)."""
    argument_shape_error = _validate_arguments_against_shape_or_error(
        mcp_tool_name, tool_arguments
    )
    if argument_shape_error:
        return (argument_shape_error, True)
    server_source_directory = (run_environment_variables or {}).get(
        "UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY"
    ) or os.environ.get("UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY")
    if not server_source_directory:
        return (
            "UNHARNESS_MCP_SERVER_SOURCE_DIRECTORY is not configured for this run — "
            "the runner cannot locate the Unharness MCP server",
            True,
        )
    try:
        import asyncio

        return asyncio.run(
            _execute_one_tools_call_over_stdio(
                mcp_tool_name,
                tool_arguments,
                server_source_directory,
                run_environment_variables or {},
            )
        )
    except Exception as execution_failure:  # noqa: BLE001 — must never crash the run
        return ("MCP tool execution failed: %r" % (execution_failure,), True)


async def _execute_one_tools_call_over_stdio(
    mcp_tool_name: str,
    tool_arguments: dict,
    server_source_directory: str,
    run_environment_variables: dict,
) -> "tuple[str, bool]":
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    child_environment = dict(os.environ)
    child_environment.update(run_environment_variables)
    child_environment["PYTHONPATH"] = os.pathsep.join(
        [server_source_directory, child_environment.get("PYTHONPATH", "")]
    )
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "unharness_mcp_server.mcp_stdio_server_entrypoint"],
        env=child_environment,
    )
    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            call_result = await session.call_tool(mcp_tool_name, tool_arguments)
            result_text = "".join(
                getattr(content_block, "text", "")
                for content_block in call_result.content
            )
            return (result_text, bool(call_result.isError))

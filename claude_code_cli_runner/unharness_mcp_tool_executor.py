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


def execute_unharness_mcp_tool_call(
    mcp_tool_name: str,
    tool_arguments: dict,
    run_environment_variables: dict,
) -> "tuple[str, bool]":
    """Spawn the Unharness MCP server over stdio and execute ONE tools/call.
    Returns ``(result_text, is_error)``."""
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

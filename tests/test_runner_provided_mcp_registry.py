"""The runner splits a session's generic assigned tool list into harness built-ins
vs runner-provided MCP names, so MCP names are held out of the harness tool flag."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner.runner_provided_mcp_registry import (
    KNOWN_RUNNER_PROVIDED_MCP_NAMES,
    UNHARNESS_API_RUNNER_MCP_NAME,
    split_assigned_tools_into_builtins_and_runner_mcps,
)


def test_unharness_api_is_a_known_runner_provided_mcp():
    assert UNHARNESS_API_RUNNER_MCP_NAME in KNOWN_RUNNER_PROVIDED_MCP_NAMES


def test_split_separates_builtins_from_runner_mcps_preserving_order():
    builtins, runner_mcps = split_assigned_tools_into_builtins_and_runner_mcps(
        ["Read", "unharness-api", "Grep"]
    )
    assert builtins == ["Read", "Grep"]
    assert runner_mcps == ["unharness-api"]


def test_split_all_builtins_yields_no_runner_mcps():
    builtins, runner_mcps = split_assigned_tools_into_builtins_and_runner_mcps(
        ["Read", "Edit"]
    )
    assert builtins == ["Read", "Edit"]
    assert runner_mcps == []


def test_split_runner_mcp_only_yields_no_builtins():
    builtins, runner_mcps = split_assigned_tools_into_builtins_and_runner_mcps(
        ["unharness-api"]
    )
    assert builtins == []
    assert runner_mcps == ["unharness-api"]

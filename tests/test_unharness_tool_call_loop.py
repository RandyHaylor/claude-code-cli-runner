"""The runner-mediated MCP tool loop (nd-472/473/474/476), against the stub harness:

- a turn whose text carries a fenced unharness-tool call has its STOP WITHHELD: the
  runner executes the (fake) tool, feeds the result back over stdin, and the agent
  continues to a genuine final turn;
- a turn with no tool call ends the run exactly as before;
- detection/translation/composition units behave.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.unharness_tool_call_detection import (
    compose_tool_result_turn_message_text,
    compose_unharness_tool_usage_instructions,
    detect_unharness_tool_calls_in_turn_text,
)
from claude_code_cli_runner.runner_provided_mcp_registry import (
    DigestibleToolNameUnknown,
    describe_digestible_tool_shapes,
    list_digestible_tool_names,
    translate_digestible_tool_name_to_mcp,
)
from conftest import stub_build_command

_FENCED_TOOL_CALL_TEXT = (
    "I need the node first.\n"
    "```unharness-tool\n"
    '{"tool": "read_one_node", "arguments": {"container_id": "nd-9"}}\n'
    "```\n"
)


# ---- detection / translation / composition units -----------------------------------

def test_detection_finds_a_fenced_tool_call_and_ignores_surrounding_text():
    calls = detect_unharness_tool_calls_in_turn_text(_FENCED_TOOL_CALL_TEXT)
    assert calls == [{"tool": "read_one_node", "arguments": {"container_id": "nd-9"}}]


def test_detection_ignores_malformed_blocks_and_plain_text():
    assert detect_unharness_tool_calls_in_turn_text("no tools here") == []
    assert detect_unharness_tool_calls_in_turn_text(
        "```unharness-tool\nnot json\n```"
    ) == []
    assert detect_unharness_tool_calls_in_turn_text(
        "```unharness-tool\n{\"arguments\": {}}\n```"  # missing tool name
    ) == []


def test_detection_finds_multiple_calls_in_order():
    two_calls = _FENCED_TOOL_CALL_TEXT + (
        "```unharness-tool\n{\"tool\": \"read_dashboard_state\"}\n```"
    )
    calls = detect_unharness_tool_calls_in_turn_text(two_calls)
    assert [c["tool"] for c in calls] == ["read_one_node", "read_dashboard_state"]
    assert calls[1]["arguments"] == {}


def test_translation_is_identity_for_known_names_and_raises_for_unknown():
    assert translate_digestible_tool_name_to_mcp("unharness-api", "read_one_node") == "read_one_node"
    # The node-building tools translate identity too (nd-473 identity seed).
    for node_building_tool in (
        "create_child_node_under_parent",
        "set_node_dependency",
        "release_node_to_pending",
        "release_children_and_return_to_pending",
    ):
        assert (
            translate_digestible_tool_name_to_mcp("unharness-api", node_building_tool)
            == node_building_tool
        )
    try:
        translate_digestible_tool_name_to_mcp("unharness-api", "no_such_tool")
        raise AssertionError("expected DigestibleToolNameUnknown")
    except DigestibleToolNameUnknown as unknown:
        assert "no_such_tool" in str(unknown)


def test_digestible_vocabulary_lists_the_unharness_tools():
    names = list_digestible_tool_names("unharness-api")
    assert "read_one_node" in names
    assert "abandon_task_so_flow_continues_without_it" in names
    # The three node-building tools are in the vocabulary, so they flow into
    # compose_unharness_tool_usage_instructions automatically (prompt cannot drift).
    assert "create_child_node_under_parent" in names
    assert "set_node_dependency" in names
    assert "release_node_to_pending" in names
    assert "release_children_and_return_to_pending" in names


def test_result_message_composition_covers_empty_and_error():
    assert "response: hi" in compose_tool_result_turn_message_text("t", "hi", False)
    assert "(no output)" in compose_tool_result_turn_message_text("t", "  ", False)
    assert "error" in compose_tool_result_turn_message_text("t", "boom", True)


def test_usage_instructions_teach_the_fence_and_the_vocabulary():
    instructions = compose_unharness_tool_usage_instructions(["read_one_node"])
    assert "```unharness-tool" in instructions
    assert "read_one_node" in instructions


def test_usage_instructions_render_parameter_shapes_when_supplied():
    shapes = [
        {
            "tool_name": "create_child_node_under_parent",
            "description": "Create ONE child node under a parent.",
            "parameters": [
                {"name": "parent_id", "type": "str", "required": True},
                {"name": "node_type", "type": "str", "required": True},
                {"name": "blocked_on", "type": "str", "required": False},
            ],
        }
    ]
    instructions = compose_unharness_tool_usage_instructions(
        ["create_child_node_under_parent"], tool_shapes=shapes
    )
    assert "create_child_node_under_parent(parent_id: str, node_type: str, blocked_on?: str)" in instructions
    assert "Create ONE child node under a parent." in instructions
    assert "EXACT argument keys" in instructions


def test_usage_instructions_fall_back_to_names_only_without_shapes():
    instructions = compose_unharness_tool_usage_instructions(
        ["read_one_node"], tool_shapes=None
    )
    assert "Available tools: read_one_node." in instructions


def test_registry_resolves_real_shapes_from_the_mcp_server_checkout():
    # The sibling checkout layout exists on the host and in the VM alike; the
    # registry imports the MCP server package's own introspection so the shapes
    # the prompt teaches can never drift from the server's registration.
    shapes = describe_digestible_tool_shapes("unharness-api")
    assert shapes, "MCP server checkout not resolvable — shapes unavailable"
    by_name = {shape["tool_name"]: shape for shape in shapes}
    create_shape = by_name["create_child_node_under_parent"]
    parameter_names = {p["name"] for p in create_shape["parameters"]}
    assert "parent_id" in parameter_names
    assert "referenced_ledger_entry_ids" in parameter_names
    # Digestible names (the prompt vocabulary) are the keys, mapped through the
    # reference file — every listed digestible name gets a shape.
    assert set(by_name) == set(list_digestible_tool_names("unharness-api"))


# ---- the loop against the stub harness ----------------------------------------------

def _run_stub_with_tool_call_turn(tmp_path, fake_executor, assigned_tool_list):
    request = RunRequest(
        input_content=[TextBlock(text="do the unharness work")],
        workspace_directory=str(tmp_path),
        assigned_tool_list=assigned_tool_list,
        unharness_tool_call_executor=fake_executor,
    )
    os.environ["STUB_TOOL_CALL_TURN_TEXT"] = _FENCED_TOOL_CALL_TEXT
    try:
        return run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_TOOL_CALL_TURN_TEXT", None)


def test_tool_call_turn_withholds_stop_executes_and_feeds_result_back(tmp_path):
    executed_calls = []

    def fake_executor(mcp_tool_name, tool_arguments, run_environment_variables):
        executed_calls.append((mcp_tool_name, tool_arguments))
        return ('{"id": "nd-9", "status": "pending"}', False)

    result = _run_stub_with_tool_call_turn(tmp_path, fake_executor, ["unharness-api"])

    # The runner executed the TRANSLATED call with the emitted arguments.
    assert executed_calls == [("read_one_node", {"container_id": "nd-9"})]
    # The agent received the tool result as its next input and finished with a
    # genuine final turn — the run's final text is turn 2, not the tool-call turn.
    assert "TOOL_RESULT_RECEIVED::" in result.assistant_text
    assert "unharness-api command submitted" in result.assistant_text
    assert '"nd-9"' in result.assistant_text
    # The withheld stop is visible in the live log as a runner note.
    with open(result.live_log_path, encoding="utf-8") as handle:
        log_text = handle.read()
    assert "stop withheld" in log_text


def test_turn_without_tool_call_ends_run_as_before(tmp_path):
    def fake_executor(mcp_tool_name, tool_arguments, run_environment_variables):
        raise AssertionError("executor must not be called when no tool call is emitted")

    request = RunRequest(
        input_content=[TextBlock(text="plain run")],
        workspace_directory=str(tmp_path),
        assigned_tool_list=["unharness-api"],
        unharness_tool_call_executor=fake_executor,
    )
    os.environ["STUB_RESULT_TEXT"] = "just a normal answer"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)
    assert result.assistant_text == "just a normal answer"


def test_tool_call_text_without_mcp_in_assigned_tools_is_not_intercepted(tmp_path):
    # No runner-provided MCP assigned => the loop is OFF: the tool-call-looking text
    # is just text; the first stop ends the run (the stub then times out waiting for
    # an injection and its extra turn is never consumed).
    def fake_executor(mcp_tool_name, tool_arguments, run_environment_variables):
        raise AssertionError("executor must not be called when the loop is disabled")

    request = RunRequest(
        input_content=[TextBlock(text="do the unharness work")],
        workspace_directory=str(tmp_path),
        assigned_tool_list=["Read"],
        unharness_tool_call_executor=fake_executor,
    )
    os.environ["STUB_TOOL_CALL_TURN_TEXT"] = _FENCED_TOOL_CALL_TEXT
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_TOOL_CALL_TURN_TEXT", None)
    assert "TOOL_RESULT_RECEIVED" not in (result.assistant_text or "")


def test_unknown_digestible_tool_is_reported_back_as_error_not_crash(tmp_path):
    def fake_executor(mcp_tool_name, tool_arguments, run_environment_variables):
        raise AssertionError("unknown tool must not reach the executor")

    request = RunRequest(
        input_content=[TextBlock(text="do the unharness work")],
        workspace_directory=str(tmp_path),
        assigned_tool_list=["unharness-api"],
        unharness_tool_call_executor=fake_executor,
    )
    os.environ["STUB_TOOL_CALL_TURN_TEXT"] = (
        "```unharness-tool\n{\"tool\": \"no_such_tool\"}\n```"
    )
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_TOOL_CALL_TURN_TEXT", None)
    # The agent still got a next-turn message (an error one) and finished.
    assert "TOOL_RESULT_RECEIVED::" in result.assistant_text
    assert "error" in result.assistant_text
    assert "no_such_tool" in result.assistant_text

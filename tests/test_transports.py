"""local vs vm_over_ssh build_command selection — location is config, not code."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner.request import RunRequest, SshConfig
from claude_code_cli_runner import transports


def test_local_build_command_runs_claude_directly():
    request = RunRequest(
        input_content=[],
        workspace_directory="/tmp/ws",
        execution_location="local_subprocess",
        claude_command="claude",
    )
    argv = transports.build_command_for(request)
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "stream-json" in argv
    assert "ssh" not in argv


def test_vm_over_ssh_build_command_wraps_in_ssh():
    request = RunRequest(
        input_content=[],
        workspace_directory="/tmp/ws",
        execution_location="vm_over_ssh",
        ssh=SshConfig(host="10.0.0.5", user="agent-user", key_path="/key"),
        dangerously_skip_permissions=True,
    )
    argv = transports.build_command_for(request)
    assert argv[0] == "ssh"
    assert "-i" in argv and "/key" in argv
    assert "agent-user@10.0.0.5" in argv
    # the remote command string carries the claude invocation
    assert any("claude" in part and "stream-json" in part for part in argv)
    assert any("--dangerously-skip-permissions" in part for part in argv)


def test_ssh_host_resolved_from_vm_name_seam(monkeypatch):
    monkeypatch.setattr(
        transports, "_vm_ip_from_dhcp_leases", lambda name: "192.168.122.99"
    )
    request = RunRequest(
        input_content=[],
        workspace_directory="/tmp/ws",
        execution_location="vm_over_ssh",
        ssh=SshConfig(vm_name="claude-vm", user="agent-user", key_path="/key"),
    )
    argv = transports.build_command_for(request)
    assert "agent-user@192.168.122.99" in argv


def test_priming_argv_is_simple_completing_call():
    request = RunRequest(input_content=[], workspace_directory="/tmp/ws")
    argv = transports.build_priming_claude_argv(request, "primed-xyz", "chunk text here")
    # Plain --session-id + positional -p prompt; NO resume/fork.
    assert "--session-id" in argv
    assert argv[argv.index("--session-id") + 1] == "primed-xyz"
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "chunk text here"
    assert "--resume" not in argv
    assert "--fork-session" not in argv
    # NOT the streaming form: it must complete on its own.
    assert "--input-format" not in argv
    assert "--output-format" not in argv
    assert "--include-partial-messages" not in argv
    assert "--verbose" not in argv
    assert "stream-json" not in argv


def test_fork_argv_resumes_primed_and_forks_without_choosing_the_id():
    # raw-1229: fork WITHOUT --session-id; claude mints the forked id, we capture it.
    request = RunRequest(input_content=[], workspace_directory="/tmp/ws")
    argv = transports.build_fork_claude_argv(request, "primed-xyz")
    assert argv[argv.index("--resume") + 1] == "primed-xyz"
    assert "--fork-session" in argv
    assert "--session-id" not in argv


def test_model_flag_threaded_into_both_transports():
    local = transports.build_command_for(
        RunRequest(input_content=[], workspace_directory="/tmp/ws", model="some-model")
    )
    assert "--model" in local and "some-model" in local


def test_permission_mode_adds_flag_and_omits_skip_permissions():
    # raw-538: an explicit permission posture launches the run at
    # --permission-mode <mode> and WITHOUT --dangerously-skip-permissions, even
    # when skip was also requested (the explicit posture wins).
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            dangerously_skip_permissions=True,
            permission_mode="acceptEdits",
        )
    )
    assert "--permission-mode" in argv
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert "--dangerously-skip-permissions" not in argv


def test_no_permission_mode_keeps_skip_permissions_path():
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            dangerously_skip_permissions=True,
        )
    )
    assert "--dangerously-skip-permissions" in argv
    assert "--permission-mode" not in argv


def _settings_json_from_argv(argv):
    import json as _json
    return _json.loads(argv[argv.index("--settings") + 1])


def test_claude_settings_overrides_emitted_as_inline_settings():
    # raw-1252: effort/thinking policy is passed as an inline --settings argument so
    # it reaches the claude process on host AND remote/VM regardless of cwd.
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            dangerously_skip_permissions=True,
            claude_settings_overrides={"effortLevel": "low", "alwaysThinkingEnabled": False},
        )
    )
    assert "--settings" in argv
    settings = _settings_json_from_argv(argv)
    assert settings["effortLevel"] == "low"
    assert settings["alwaysThinkingEnabled"] is False


def test_settings_overrides_merged_with_permission_mode_default_posture():
    # The overrides and the permission-mode default posture are MERGED into one
    # --settings argument, not emitted twice.
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            permission_mode="acceptEdits",
            claude_settings_overrides={"effortLevel": "low", "alwaysThinkingEnabled": False},
        )
    )
    assert argv.count("--settings") == 1
    settings = _settings_json_from_argv(argv)
    assert settings["effortLevel"] == "low"
    assert settings["alwaysThinkingEnabled"] is False
    assert settings["permissions"]["defaultMode"] == "acceptEdits"


def test_no_overrides_and_no_permission_mode_omits_settings():
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            dangerously_skip_permissions=True,
        )
    )
    assert "--settings" not in argv


def test_explicit_session_id_creates_session():
    # raw-538 resume-on-reply turn 1: --session-id <id> (create), not --resume.
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            session_id="sess-123",
        )
    )
    assert argv[argv.index("--session-id") + 1] == "sess-123"
    assert "--resume" not in argv


def test_permission_mode_adds_permission_prompt_tool_stdio():
    # raw-538/nd-251: a permission posture drives the can_use_tool control
    # protocol over stdio, so --permission-prompt-tool stdio must be present.
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            permission_mode="acceptEdits",
        )
    )
    assert "--permission-prompt-tool" in argv
    assert argv[argv.index("--permission-prompt-tool") + 1] == "stdio"
    # The posture is pinned via --settings so a host/VM bypass default cannot
    # leak through and skip the gate.
    assert "--settings" in argv
    settings_json = argv[argv.index("--settings") + 1]
    assert '"defaultMode": "acceptEdits"' in settings_json


def test_no_permission_mode_omits_permission_prompt_tool():
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            dangerously_skip_permissions=True,
        )
    )
    assert "--permission-prompt-tool" not in argv


def test_resume_session_continues_existing_session():
    # raw-538 resume-on-reply later turn: --resume <id> (continue same chat).
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[],
            workspace_directory="/tmp/ws",
            session_id="sess-123",
            resume_session=True,
        )
    )
    assert argv[argv.index("--resume") + 1] == "sess-123"
    assert "--session-id" not in argv


def test_no_assigned_tool_list_omits_tools_flag():
    # None assigned list => leave the default toolset unchanged (no --tools emitted).
    argv = transports.build_base_claude_argv(
        RunRequest(input_content=[], workspace_directory="/tmp/ws")
    )
    assert "--tools" not in argv


def test_assigned_tools_without_whitelist_do_not_restrict():
    # A list present but NOT enforced as a whitelist => built-ins are not restricted,
    # so no --tools is emitted (the harness keeps its default toolset).
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[], workspace_directory="/tmp/ws",
            assigned_tool_list=["Read"],
            restrict_to_assigned_tools_as_whitelist=False,
        )
    )
    assert "--tools" not in argv


def test_whitelist_with_empty_builtins_disables_all_local_tools():
    # Whitelist true + no built-ins => a prose-only, no-local-tools session: --tools ""
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[], workspace_directory="/tmp/ws",
            assigned_tool_list=[],
            restrict_to_assigned_tools_as_whitelist=True,
        )
    )
    assert argv[argv.index("--tools") + 1] == ""


def test_whitelist_restricts_to_assigned_builtin_tools():
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[], workspace_directory="/tmp/ws",
            assigned_tool_list=["Read", "Grep"],
            restrict_to_assigned_tools_as_whitelist=True,
        )
    )
    assert argv[argv.index("--tools") + 1] == "Read,Grep"


def test_whitelist_holds_runner_mcp_names_out_of_the_builtin_tool_flag():
    # A runner-provided MCP name (unharness-api) is NOT a harness built-in, so it must
    # NOT be passed to --tools; only the built-in tools are (the MCP is enabled via the
    # runner's MCP-client layer instead).
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[], workspace_directory="/tmp/ws",
            assigned_tool_list=["Read", "unharness-api"],
            restrict_to_assigned_tools_as_whitelist=True,
        )
    )
    assert argv[argv.index("--tools") + 1] == "Read"


def test_api_only_posture_is_no_builtins_plus_runner_mcp():
    # The API-only agent: whitelist true, only a runner-MCP assigned => zero built-in
    # tools (--tools "") while the runner-MCP is held out for the MCP layer.
    argv = transports.build_base_claude_argv(
        RunRequest(
            input_content=[], workspace_directory="/tmp/ws",
            assigned_tool_list=["unharness-api"],
            restrict_to_assigned_tools_as_whitelist=True,
        )
    )
    assert argv[argv.index("--tools") + 1] == ""

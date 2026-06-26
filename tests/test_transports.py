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


def test_fork_argv_resumes_primed_and_forks_to_fresh():
    request = RunRequest(input_content=[], workspace_directory="/tmp/ws")
    argv = transports.build_fork_claude_argv(request, "primed-xyz", "task-abc")
    assert argv[argv.index("--resume") + 1] == "primed-xyz"
    assert "--fork-session" in argv
    assert argv[argv.index("--session-id") + 1] == "task-abc"


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

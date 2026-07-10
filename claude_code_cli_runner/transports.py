"""Transports: build the argv for a streaming `claude -p`, locally or over SSH.

PROVING "location is config, not code": every execution_location resolves to a
``build_command() -> argv`` callable, and they ALL feed the same streaming
runner. local_subprocess runs claude directly; vm_over_ssh / remote_host wrap
the same claude argv in an ``ssh`` invocation. The prompt is NEVER on argv (real
claude in --input-format stream-json mode waits for a stdin user message), so it
never lands on a process table — for local or remote runs alike.
"""

from __future__ import annotations

import json
import shlex
import subprocess

from .request import (
    HARNESS_CLAUDE_CLI,
    HARNESS_OPENCODE_CLI,
    LOCATION_LOCAL_SUBPROCESS,
    LOCATION_REMOTE_HOST,
    LOCATION_VM_OVER_SSH,
    RunRequest,
)
from .runner_provided_mcp_registry import (
    split_assigned_tools_into_builtins_and_runner_mcps,
)
from .harness_tool_name_translation import (
    translate_generic_tool_names_to_harness_specific,
)


def build_base_claude_argv(run_request: RunRequest) -> "list[str]":
    """The verified streaming claude argv (no prompt positional).

    stream-json out with partial messages + stream-json in (so the prompt and
    any send_command can be injected over stdin) + --verbose (required for
    stream-json output in -p mode).
    """
    argv = [
        run_request.claude_command,
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--input-format",
        "stream-json",
        "--verbose",
    ]
    # An explicit permission posture WINS over full bypass: a collaborative /
    # manual task launches at --permission-mode <mode> and is NOT given
    # --dangerously-skip-permissions, so the agent is not fully unattended.
    if run_request.permission_mode:
        argv[2:2] = ["--permission-mode", run_request.permission_mode]
        # Live tool-permission escalation: with a permission posture set, drive the
        # CLI's can_use_tool control protocol over stdio so tools needing approval
        # emit a request the runner can answer (allow/deny) — verified flag on the
        # claude CLI (real but not shown in --help). The runner sends the
        # initialize handshake + control_response decisions.
        argv += ["--permission-prompt-tool", "stdio"]
    elif run_request.dangerously_skip_permissions:
        argv.insert(2, "--dangerously-skip-permissions")
    # Compose ONE inline --settings argument (raw-1252): the caller's claude settings
    # overrides (effort/thinking, etc.) MERGED with the permission-mode default posture.
    # Inline --settings applies on the host AND a remote/VM claude regardless of cwd and
    # OUTRANKS the environment's settings.json (closing the gap where a per-workspace
    # settings.local.json is not read by a remote claude). Emitted only when non-empty.
    combined_claude_settings = dict(run_request.claude_settings_overrides or {})
    if run_request.permission_mode:
        # Pin the session's default permission posture so a manual/collaborative run
        # actually GATES — otherwise a host/VM standing ``defaultMode:
        # bypassPermissions`` leaks through and auto-allows tools (verified on the
        # sandbox VM), defeating the prompt.
        permissions_block = dict(combined_claude_settings.get("permissions") or {})
        permissions_block["defaultMode"] = run_request.permission_mode
        combined_claude_settings["permissions"] = permissions_block
    if combined_claude_settings:
        argv += ["--settings", json.dumps(combined_claude_settings, sort_keys=True)]
    # Caller session-start steering (raw-1255): append text to the system prompt via
    # the official --append-system-prompt flag. Part of the (stable) system prompt, so
    # it stays in the cached prefix shared by warm forks. Emitted only when non-empty.
    if run_request.append_system_prompt_text:
        argv += ["--append-system-prompt", run_request.append_system_prompt_text]
    if run_request.model:
        argv[2:2] = ["--model", run_request.model]
    # Explicit session id so the session is resumable across turns (resume-on-
    # reply, collaborative tasks). resume_session => CONTINUE the existing
    # session; otherwise CREATE it with this id. (The prime/fork reuse path uses
    # its own argv builders and never sets run_request.session_id, so no clash.)
    if run_request.session_id:
        if run_request.resume_session:
            argv += ["--resume", run_request.session_id]
        else:
            argv += ["--session-id", run_request.session_id]
    # Per-session tool restriction. When the session's assigned tool list is enforced
    # as an exclusive whitelist, restrict the harness to ONLY the assigned BUILT-IN
    # tools via --tools (the runner-provided MCP names are held out — they are enabled
    # through the runner's MCP-client layer, not the harness tool flag; and --tools
    # governs built-ins only, per the CLI docs). Verified on the installed CLI:
    # `--tools ""` disables all built-ins, `--tools "Bash,Edit,Read"` limits to those.
    # (This is distinct from --allowedTools, which only pre-approves.) Names are
    # comma-joined into one argument so the flag's greedy nargs cannot swallow later
    # flags. Not restricting => emit no --tools (harness default toolset).
    if (
        run_request.restrict_to_assigned_tools_as_whitelist
        and run_request.assigned_tool_list is not None
    ):
        generic_builtin_tool_names, _runner_mcp_names = (
            split_assigned_tools_into_builtins_and_runner_mcps(
                run_request.assigned_tool_list
            )
        )
        # The assigned names are GENERIC (Unharness-level); convert to claude-specific
        # names before emitting --tools (the runner owns this harness translation).
        claude_specific_tool_names = translate_generic_tool_names_to_harness_specific(
            generic_builtin_tool_names, HARNESS_CLAUDE_CLI
        )
        argv += ["--tools", ",".join(claude_specific_tool_names)]
    argv.extend(run_request.extra_cli_flags)
    return argv


def build_base_opencode_argv(run_request: RunRequest) -> "list[str]":
    """The streaming opencode argv (no prompt positional).

    ``opencode run --format json`` emits raw JSON events; the prompt is
    delivered over stdin (verified: opencode reads the message from stdin when
    no positional is given), so it never lands on a process table — the same
    privacy contract as the claude argv.
    """
    argv = [
        run_request.opencode_command,
        "run",
        "--format",
        "json",
        # Emit thinking/reasoning blocks as stream events. Some local models
        # (observed: ollama gemma4:e4b) put their ENTIRE answer in a reasoning
        # part and stop; without this flag opencode emits no event for it and
        # the run looks blank.
        "--thinking",
    ]
    if run_request.dangerously_skip_permissions:
        # opencode's full-auto switch: auto-approve anything not explicitly
        # denied. The sandbox/VM boundary is the safety layer, exactly as with
        # claude --dangerously-skip-permissions.
        argv.append("--auto")
    if run_request.model:
        argv += ["--model", run_request.model]
    if run_request.session_id and run_request.resume_session:
        argv += ["--session", run_request.session_id]
    argv.extend(run_request.extra_cli_flags)
    return argv


def build_base_harness_argv(run_request: RunRequest) -> "list[str]":
    """The ONLY place the harness choice branches into an argv builder."""
    if run_request.harness == HARNESS_OPENCODE_CLI:
        return build_base_opencode_argv(run_request)
    return build_base_claude_argv(run_request)


def build_priming_claude_argv(
    run_request: RunRequest, primed_session_id: str, chunk_text: str
) -> "list[str]":
    """Argv for a PRIMING run: a SIMPLE, self-completing ``claude -p`` invocation
    that creates a NEW session with a caller-chosen id (``--session-id``) and
    ingests the chunk as a positional prompt.

    VERIFIED against real claude: the streaming base argv (stream-json in/out,
    chunk over stdin) does NOT complete a priming session — no result is ever
    produced, so priming always failed and reuse fell back to inline. A plain
    completing call DOES persist a forkable primed session:

        claude [--model M] [--dangerously-skip-permissions] \\
               --session-id <primed_sid> -p "<chunk text>"

    NO --output-format/--input-format/--include-partial-messages/--verbose and
    NO stdin: the prompt is the positional arg, and claude exits 0 on its own.
    Later task runs fork this session (the chunk is then a cache read).
    """
    argv = [run_request.claude_command]
    if run_request.model:
        argv += ["--model", run_request.model]
    if run_request.dangerously_skip_permissions:
        argv += ["--dangerously-skip-permissions"]
    argv += ["--session-id", primed_session_id, "-p", chunk_text]
    return argv


def build_fork_claude_argv(
    run_request: RunRequest, primed_session_id: str
) -> "list[str]":
    """Argv for a TASK run that FORKS from an already-primed session.

    ``claude --resume <primed> --fork-session`` creates a new session that inherits
    the primed session's history; the primed session is untouched and reusable. The
    NEW forked session id is MINTED BY CLAUDE and reported on the run's result event —
    we do NOT pass ``--session-id`` here. Combining ``--session-id`` with
    ``--fork-session`` is undocumented and behaved inconsistently live (claude sometimes
    left the passed id a near-empty stub and wrote the work to its own minted id,
    raw-1224..1229). So the caller CAPTURES claude's minted fork id, exactly as it
    captures opencode's minted id. The per-task input is delivered over stdin as usual.
    """
    argv = build_base_claude_argv(run_request)
    argv += ["--resume", primed_session_id, "--fork-session"]
    return argv


def resolve_ssh_host(ssh_config) -> str:
    """Resolve the SSH host: an explicit host wins; otherwise look the VM up by
    name via libvirt DHCP leases. This is the single real-infra seam — tests
    monkeypatch it so the SSH path runs against a stub with no real VM."""
    if ssh_config.host:
        return ssh_config.host
    if ssh_config.vm_name:
        return _vm_ip_from_dhcp_leases(ssh_config.vm_name)
    raise ValueError("ssh config needs either 'host' or 'vm_name'")


def _vm_ip_from_dhcp_leases(vm_name: str) -> str:
    """Resolve a VM's IP from libvirt DHCP leases (optional convenience)."""
    completed = subprocess.run(
        ["virsh", "net-dhcp-leases", "default"],
        capture_output=True,
        text=True,
        check=True,
    )
    ip = ""
    for line in completed.stdout.splitlines():
        if vm_name in line:
            for field in line.split():
                if "/" in field and field.split("/")[0].count(".") == 3:
                    ip = field.split("/")[0]
    if not ip:
        raise ValueError("could not determine IP for VM %r from DHCP leases" % vm_name)
    return ip


def build_ssh_argv(run_request: RunRequest) -> "list[str]":
    """Wrap the base claude argv in an ssh invocation to the configured host.

    The remote command cd's into the remote workspace (if given) then runs the
    shell-quoted claude argv. SSH forwards the host process's stdin straight to
    remote claude's stdin, so prompt delivery + send_command injection work
    identically to a local run — just one hop further.
    """
    ssh_config = run_request.ssh
    if ssh_config is None:
        raise ValueError("ssh execution_location requires an ssh config")
    host = resolve_ssh_host(ssh_config)

    remote_argv = build_base_harness_argv(run_request)
    remote_command = " ".join(shlex.quote(part) for part in remote_argv)
    if ssh_config.remote_workspace_directory:
        remote_command = (
            "cd " + shlex.quote(ssh_config.remote_workspace_directory) + "; " + remote_command
        )

    ssh_argv = ["ssh"]
    if ssh_config.key_path:
        ssh_argv += ["-i", ssh_config.key_path]
    if ssh_config.port and ssh_config.port != 22:
        ssh_argv += ["-p", str(ssh_config.port)]
    ssh_argv += [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "%s@%s" % (ssh_config.user, host),
        remote_command,
    ]
    return ssh_argv


def build_command_for(run_request: RunRequest) -> "list[str]":
    """Select and build the argv for the request's execution_location.

    The ONLY place location branches. Everything downstream (the streaming
    runner) is identical regardless of which argv this returns.
    """
    location = run_request.execution_location
    if location == LOCATION_LOCAL_SUBPROCESS:
        return build_base_harness_argv(run_request)
    if location in (LOCATION_VM_OVER_SSH, LOCATION_REMOTE_HOST):
        return build_ssh_argv(run_request)
    raise ValueError("unknown execution_location %r" % location)

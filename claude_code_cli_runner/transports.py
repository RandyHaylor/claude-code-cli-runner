"""Transports: build the argv for a streaming `claude -p`, locally or over SSH.

PROVING "location is config, not code": every execution_location resolves to a
``build_command() -> argv`` callable, and they ALL feed the same streaming
runner. local_subprocess runs claude directly; vm_over_ssh / remote_host wrap
the same claude argv in an ``ssh`` invocation. The prompt is NEVER on argv (real
claude in --input-format stream-json mode waits for a stdin user message), so it
never lands on a process table — for local or remote runs alike.
"""

from __future__ import annotations

import shlex
import subprocess

from .request import (
    LOCATION_LOCAL_SUBPROCESS,
    LOCATION_REMOTE_HOST,
    LOCATION_VM_OVER_SSH,
    RunRequest,
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
    if run_request.dangerously_skip_permissions:
        argv.insert(2, "--dangerously-skip-permissions")
    if run_request.model:
        argv[2:2] = ["--model", run_request.model]
    argv.extend(run_request.extra_cli_flags)
    return argv


def build_priming_claude_argv(run_request: RunRequest, primed_session_id: str) -> "list[str]":
    """Argv for a PRIMING run: a normal streaming claude that creates a NEW
    session with a caller-chosen id (``--session-id``). The reusable context
    chunk is delivered over stdin as the user message; the resulting session is
    recorded so later tasks can fork from it.

    Same verified base argv as a normal run, plus a plain ``--session-id`` so
    the session has a stable, reusable id.
    """
    argv = build_base_claude_argv(run_request)
    argv += ["--session-id", primed_session_id]
    return argv


def build_fork_claude_argv(
    run_request: RunRequest, primed_session_id: str, task_session_id: str
) -> "list[str]":
    """Argv for a TASK run that FORKS from an already-primed session.

    ``claude --resume <primed> --fork-session --session-id <fresh>`` creates a
    new session that inherits the primed session's history (the chunk is already
    in it); the primed session is untouched and reusable. The per-task remainder
    (``input_content``) is delivered over stdin as usual.
    """
    argv = build_base_claude_argv(run_request)
    argv += [
        "--resume",
        primed_session_id,
        "--fork-session",
        "--session-id",
        task_session_id,
    ]
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

    remote_argv = build_base_claude_argv(run_request)
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
        return build_base_claude_argv(run_request)
    if location in (LOCATION_VM_OVER_SSH, LOCATION_REMOTE_HOST):
        return build_ssh_argv(run_request)
    raise ValueError("unknown execution_location %r" % location)

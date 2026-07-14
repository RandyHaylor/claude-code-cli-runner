"""Transports: choose WHERE a harness runs (execution location) and build the
launch command accordingly. The WHAT (per-harness argv) is owned by each harness
integration module; this file asks the harness integration for the base argv and
only adds the location wrapping (local subprocess vs ssh to a VM / remote host).

PROVING "location is config, not code": every execution_location resolves to a
``build_command() -> argv`` callable, and they ALL feed the same streaming
runner. local_subprocess runs the harness directly; vm_over_ssh / remote_host
wrap the same base argv in an ``ssh`` invocation.
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


def build_base_harness_argv(run_request: RunRequest) -> "list[str]":
    """Build the launch argv by asking the request's harness integration — no
    harness conditional here; each harness owns its own argv construction."""
    from .harness_integration import get_harness_integration

    return get_harness_integration(run_request.harness).build_launch_command(run_request)


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
    """Wrap the base harness argv in an ssh invocation to the configured host.

    The remote command cd's into the remote workspace (if given) then runs the
    shell-quoted harness argv. SSH forwards the host process's stdin straight to
    the remote harness's stdin, so prompt delivery + send_command injection work
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

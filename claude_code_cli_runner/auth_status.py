"""Check whether the local ``claude`` CLI is authenticated (logged in).

Runs ``claude auth status --json`` in THIS machine's environment and normalizes
its result into a small, stable dict. The runner exposes this so a caller can
PRE-FLIGHT the harness login state — and surface "needs ``claude auth login``" —
without dispatching a real task that would fail with a 401 authentication error.

The runner is location-agnostic: each runner instance checks its OWN local
``claude`` credentials. To check a remote machine's harness (e.g. the in-VM
runner), call THAT runner's serve face; no SSH is performed here.

``claude auth status --json`` emits (verified against the real CLI)::

    {"loggedIn": true, "authMethod": "claude.ai", "email": "...",
     "orgId": "...", "orgName": "...", "subscriptionType": "team"}

The authoritative signal is the ``loggedIn`` boolean; the rest is descriptive.
"""

from __future__ import annotations

import json
import subprocess

# The dedicated, non-prompting, non-billing auth check the claude CLI provides.
CLAUDE_AUTH_STATUS_COMMAND = ["claude", "auth", "status", "--json"]


def check_claude_auth_status(run_subprocess=subprocess.run, command=None) -> dict:
    """Run ``claude auth status --json`` locally and return a normalized status.

    ``run_subprocess`` is injectable (tests pass a stub so no real ``claude`` is
    invoked). Returns a dict that ALWAYS carries ``logged_in`` (bool) and
    ``error`` (str or None), plus, when available, ``email``/``org_name``/
    ``subscription_type``/``auth_method``, the process ``exit_code`` and the raw
    stdout/stderr. A missing CLI, timeout, or unparseable output is reported as
    ``logged_in=False`` with a descriptive ``error`` — never raised.
    """
    argv = list(command) if command is not None else list(CLAUDE_AUTH_STATUS_COMMAND)
    try:
        completed = run_subprocess(argv, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as cause:
        return _failure("claude CLI not found: %s" % cause)
    except subprocess.TimeoutExpired:
        return _failure("`claude auth status` timed out")
    except Exception as cause:  # never let a probe raise into the caller
        return _failure("failed to run `claude auth status`: %s" % cause)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    status = {
        "exit_code": completed.returncode,
        "raw_stdout": stdout,
        "raw_stderr": stderr,
        "error": None,
    }
    try:
        parsed = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        status["logged_in"] = False
        status["error"] = "could not parse `claude auth status --json` output"
        return status

    status["logged_in"] = bool(parsed.get("loggedIn"))
    for source_key, destination_key in (
        ("email", "email"),
        ("orgName", "org_name"),
        ("orgId", "org_id"),
        ("subscriptionType", "subscription_type"),
        ("authMethod", "auth_method"),
    ):
        if source_key in parsed:
            status[destination_key] = parsed[source_key]
    return status


def _failure(error_message: str) -> dict:
    return {
        "logged_in": False,
        "exit_code": None,
        "raw_stdout": "",
        "raw_stderr": "",
        "error": error_message,
    }

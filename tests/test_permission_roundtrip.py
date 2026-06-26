"""raw-538/nd-251: end-to-end (stub) round-trip of the can_use_tool control
protocol. The stub emits a permission request; the runner surfaces it as a
permission_request live-log record and HOLDS awaiting a decision; we write a
permission_decision control intent; the runner relays the control_response and
the stub reports back the decision it received."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.live_files import (
    CONTROL_PERMISSION_DECISION,
    append_control_intent,
    live_log_path,
)
from conftest import stub_build_command


def _run_with_permission_decision(tmp_path, behavior):
    os.environ["STUB_REQUEST_PERMISSION"] = "1"
    holder = {}

    def run_it():
        request = RunRequest(
            input_content=[TextBlock(text="do something needing permission")],
            workspace_directory=str(tmp_path),
            permission_mode="acceptEdits",  # enables the control protocol in the runner
        )
        holder["result"] = run_claude_code_task(request, build_command=stub_build_command)

    worker = threading.Thread(target=run_it)
    worker.start()
    try:
        # Wait until the runner has surfaced the permission request, then decide.
        deadline = time.time() + 8
        log_path = live_log_path(str(tmp_path))
        saw_request = False
        while time.time() < deadline and not saw_request:
            if os.path.isfile(log_path):
                with open(log_path, encoding="utf-8") as handle:
                    if "permission_request" in handle.read():
                        saw_request = True
                        break
            time.sleep(0.05)
        assert saw_request, "runner never surfaced the permission_request record"
        append_control_intent(
            str(tmp_path),
            CONTROL_PERMISSION_DECISION,
            decision={"request_id": "stub-perm-1", "behavior": behavior},
        )
    finally:
        worker.join(timeout=10)
        os.environ.pop("STUB_REQUEST_PERMISSION", None)
    return holder["result"]


def _live_log_text(tmp_path):
    with open(live_log_path(str(tmp_path)), encoding="utf-8") as handle:
        return handle.read()


def test_allow_decision_round_trips_to_the_agent(tmp_path):
    result = _run_with_permission_decision(tmp_path, "allow")
    assert "PERMISSION:allow" in result.assistant_text
    # The decision is marked resolved in the live log (the host-visible signal the
    # dashboard uses to stop showing the request as actionable).
    log = _live_log_text(tmp_path)
    assert "permission_request" in log
    assert "permission_resolved" in log


def test_deny_decision_round_trips_to_the_agent(tmp_path):
    result = _run_with_permission_decision(tmp_path, "deny")
    assert "PERMISSION:deny" in result.assistant_text

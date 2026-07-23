"""A harness that stays ALIVE but goes SILENT after an abort must not hang the
run: the post-abort final-output read is bounded by a wall deadline at the
READ level (observed live 2026-07-23 — pi never acked the abort and the run
thread blocked forever on next())."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.live_files import (
    CONTROL_END_AND_RETURN,
    append_control_intent,
)

_SILENT_AFTER_ABORT_PI_STUB = r'''
import json, sys, time
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
emit({"type": "session", "id": "sess-silent"})
emit({"type": "agent_start"})
emit({"type": "turn_start"})
for index in range(200):
    emit({"type": "message_update",
          "assistantMessageEvent": {"type": "text_delta", "delta": "word%d " % index}})
    time.sleep(0.05)
# never emits turn_end/agent_end, never acks abort/get_session_stats,
# and STAYS ALIVE silently (reads stdin forever without responding).
for _line in sys.stdin:
    pass
time.sleep(600)
'''


def test_run_ends_promptly_when_harness_goes_silent_after_abort(tmp_path):
    stub_path = tmp_path / "silent_after_abort_pi_stub.py"
    stub_path.write_text(_SILENT_AFTER_ABORT_PI_STUB, encoding="utf-8")

    def build_silent_pi_stub_command(run_request):
        return [sys.executable, str(stub_path)]

    def queue_end_and_return_shortly():
        time.sleep(0.5)
        append_control_intent(str(tmp_path), CONTROL_END_AND_RETURN)

    stopper = threading.Thread(target=queue_end_and_return_shortly)
    stopper.start()
    started_at = time.monotonic()
    try:
        result = run_claude_code_task(
            RunRequest(
                input_content=[TextBlock(text="go")],
                workspace_directory=str(tmp_path),
                harness="pi",
            ),
            build_command=build_silent_pi_stub_command,
        )
    finally:
        stopper.join()
    elapsed_seconds = time.monotonic() - started_at

    assert result.operator_ended is True
    # 5s post-abort budget + margins; the pre-fix behavior hung forever.
    assert elapsed_seconds < 15
    assert result.harness_session_id == "sess-silent"

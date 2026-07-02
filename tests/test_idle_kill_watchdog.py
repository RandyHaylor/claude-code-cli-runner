"""The runner's idle-kill duty: when the harness process streams NOTHING for
idle_kill_seconds, the RUNNER kills it (the caller only supplies the policy
value). The result reports run_state idle_killed, the live log records the
kill, and the harness process is really dead."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task

_SILENT_STUB_SOURCE = """\
import sys, time
sys.stdin.readline()  # consume the prompt like a real harness
print('{"type": "assistant", "message": {"content": [{"type": "text", '
      '"text": "starting"}]}}', flush=True)
time.sleep(120)  # go SILENT far past any idle budget
"""


def _silent_stub_build_command(tmp_path):
    silent_stub_path = os.path.join(str(tmp_path), "stub_silent_claude.py")
    with open(silent_stub_path, "w", encoding="utf-8") as handle:
        handle.write(_SILENT_STUB_SOURCE)

    def build_command(run_request):
        return [sys.executable, silent_stub_path]

    return build_command


def test_silent_harness_is_killed_within_the_idle_budget(tmp_path):
    request = RunRequest(
        input_content=[TextBlock(text="do something")],
        workspace_directory=str(tmp_path),
        idle_kill_seconds=1.5,
    )
    started = time.monotonic()
    result = run_claude_code_task(
        request, build_command=_silent_stub_build_command(tmp_path)
    )
    elapsed = time.monotonic() - started

    assert elapsed < 30, "the watchdog must end the run long before the stub's sleep"
    assert result.run_state == "idle_killed"
    assert "no stream activity" in (result.harness_stderr or "")
    with open(result.live_log_path, encoding="utf-8") as handle:
        log_lines = [json.loads(line) for line in handle if line.strip()]
    assert any("runner_idle_kill" in record for record in log_lines)


def test_streaming_harness_is_not_killed(tmp_path):
    from conftest import stub_build_command

    request = RunRequest(
        input_content=[TextBlock(text="hello stub")],
        workspace_directory=str(tmp_path),
        idle_kill_seconds=30,
    )
    os.environ["STUB_RESULT_TEXT"] = "finished normally"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)
    assert result.run_state != "idle_killed"
    assert result.assistant_text == "finished normally"

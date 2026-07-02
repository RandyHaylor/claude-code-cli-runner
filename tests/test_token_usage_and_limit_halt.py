"""Token accounting + the hard token-limit halt (the runner's duty; the caller
only supplies task_token_limit): usage is ALWAYS collected from the stream's
message_delta usage objects and reported on the result; when counted tokens
(input + output + cache-creation; cache reads excluded) reach the limit, the
runner halts the harness immediately and reports the coded halt."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task

# Streams two turns, each with a usage-bearing message_delta (500 counted
# tokens per turn: 100 in + 300 out + 100 cache-creation; 1000 cache reads are
# NOT counted), then would sleep long — a limit must halt it first.
_USAGE_STREAMING_STUB_SOURCE = """\
import sys, time

def emit(document):
    print(document, flush=True)

sys.stdin.readline()
for turn_number in (1, 2):
    emit('{"type": "assistant", "message": {"content": [{"type": "text", '
         '"text": "turn %d "}]}}' % turn_number)
    emit('{"type": "stream_event", "event": {"type": "message_delta", '
         '"usage": {"input_tokens": 100, "output_tokens": 300, '
         '"cache_creation_input_tokens": 100, '
         '"cache_read_input_tokens": 1000}}}')
    time.sleep(0.1)
time.sleep(60)
"""


def _usage_stub_build_command(tmp_path):
    stub_path = os.path.join(str(tmp_path), "stub_usage_streaming_claude.py")
    with open(stub_path, "w", encoding="utf-8") as handle:
        handle.write(_USAGE_STREAMING_STUB_SOURCE)

    def build_command(run_request):
        return [sys.executable, stub_path]

    return build_command


def test_limit_reached_halts_the_run_and_reports_the_coded_halt(tmp_path):
    request = RunRequest(
        input_content=[TextBlock(text="burn tokens")],
        workspace_directory=str(tmp_path),
        task_token_limit=800,  # first turn = 500 counted; second crosses it
    )
    started = time.monotonic()
    result = run_claude_code_task(
        request, build_command=_usage_stub_build_command(tmp_path)
    )
    elapsed = time.monotonic() - started

    assert elapsed < 30, "the halt must fire long before the stub's sleep"
    assert result.run_state == "token_limit_halted"
    assert result.token_usage["counted_tokens"] == 1000  # two 500-token turns
    assert result.token_usage["cache_read_input_tokens"] == 2000  # reported
    assert "halted due to token limit" in (result.harness_stderr or "")
    with open(result.live_log_path, encoding="utf-8") as handle:
        assert any("runner_token_limit_halt" in line for line in handle)


def test_usage_is_reported_without_any_limit(tmp_path):
    from conftest import stub_build_command

    request = RunRequest(
        input_content=[TextBlock(text="hello stub")],
        workspace_directory=str(tmp_path),
    )
    os.environ["STUB_RESULT_TEXT"] = "fine"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)
    assert result.run_state != "token_limit_halted"
    assert set(result.token_usage.keys()) == {
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens", "counted_tokens",
    }


def test_run_under_its_limit_finishes_normally(tmp_path):
    from conftest import stub_build_command

    request = RunRequest(
        input_content=[TextBlock(text="hello stub")],
        workspace_directory=str(tmp_path),
        task_token_limit=10_000_000,
    )
    os.environ["STUB_RESULT_TEXT"] = "finished under budget"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)
    assert result.run_state != "token_limit_halted"
    assert result.assistant_text == "finished under budget"

"""The standard harness-activity-event vocabulary (raw-780/781): BOTH harness
chunk shapes normalize to the same neutral kinds, and streamed runs write the
``activity`` facet on their live-log records."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.harness_activity_events import (
    derive_harness_activity_event,
)
from test_opencode_harness import (
    _opencode_request,
    stub_opencode_build_command,
)
from conftest import stub_build_command


# --- normalization: claude-shaped chunks --------------------------------------

def test_claude_streamed_text_delta_normalizes_to_assistant_text():
    # A real claude run's text arrives ONCE via partial deltas; the later full
    # assistant message must NOT re-emit it (dedup).
    activity = derive_harness_activity_event(
        {"type": "stream_event",
         "event": {"type": "content_block_delta",
                   "delta": {"type": "text_delta", "text": "hel"}}}
    )
    assert activity == {"kind": "assistant_text", "text": "hel"}
    duplicate_full_message = derive_harness_activity_event(
        {"type": "assistant",
         "message": {"content": [{"type": "text", "text": "hello"}]}}
    )
    assert duplicate_full_message is None


def test_translator_marked_assistant_text_normalizes_once():
    # The opencode translator emits text exactly once, marked opencode_event.
    activity = derive_harness_activity_event(
        {"type": "assistant", "opencode_event": {"type": "text"},
         "message": {"content": [{"type": "text", "text": "hello"}]}}
    )
    assert activity == {"kind": "assistant_text", "text": "hello"}


def test_claude_tool_use_block_normalizes_to_tool_activity():
    activity = derive_harness_activity_event(
        {"type": "assistant",
         "message": {"content": [
             {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}}
    )
    assert activity["kind"] == "tool_activity"
    assert activity["tool_name"] == "Bash"
    assert "ls" in activity["input_summary"]


def test_usage_delta_normalizes_to_turn_usage():
    activity = derive_harness_activity_event(
        {"type": "stream_event",
         "event": {"type": "message_delta", "usage": {"output_tokens": 5}}}
    )
    assert activity == {"kind": "turn_usage", "token_usage": {"output_tokens": 5}}


def test_result_normalizes_to_run_result():
    activity = derive_harness_activity_event({"type": "result", "result": "done!"})
    assert activity == {"kind": "run_result", "final_text": "done!"}


# --- normalization: opencode passthrough events --------------------------------

def test_opencode_reasoning_normalizes_to_reasoning_text():
    activity = derive_harness_activity_event(
        {"type": "reasoning", "part": {"type": "reasoning", "text": "thinking"}}
    )
    assert activity == {"kind": "reasoning_text", "text": "thinking"}


def test_opencode_tool_use_normalizes_to_tool_activity():
    activity = derive_harness_activity_event(
        {"type": "tool_use",
         "part": {"type": "tool", "tool": "grep",
                  "state": {"status": "completed",
                            "input": {"pattern": "claw"},
                            "output": "3 matches"}}}
    )
    assert activity["kind"] == "tool_activity"
    assert activity["tool_name"] == "grep"
    assert activity["status"] == "completed"
    assert "claw" in activity["input_summary"]
    assert "3 matches" in activity["output_summary"]


def test_protocol_noise_normalizes_to_none():
    assert derive_harness_activity_event({"type": "step_start", "part": {}}) is None
    assert derive_harness_activity_event("not a dict") is None


# --- end-to-end: streamed runs write the activity facet ------------------------

def _read_activity_kinds(live_log_path):
    kinds = []
    with open(live_log_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("activity"):
                kinds.append(record["activity"]["kind"])
    return kinds


def test_opencode_run_records_carry_activity_facets(tmp_path):
    result = run_claude_code_task(
        _opencode_request(tmp_path), build_command=stub_opencode_build_command
    )
    kinds = _read_activity_kinds(result.live_log_path)
    assert "assistant_text" in kinds
    assert "turn_usage" in kinds
    assert "run_result" in kinds


def test_claude_run_records_carry_activity_facets(tmp_path):
    request = RunRequest(
        input_content=[TextBlock(text="hello stub")],
        workspace_directory=str(tmp_path),
    )
    os.environ["STUB_RESULT_TEXT"] = "the answer is 42"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)
    kinds = _read_activity_kinds(result.live_log_path)
    assert "run_result" in kinds

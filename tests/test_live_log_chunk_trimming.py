"""The live log must not balloon when a harness streams a long message/tool-call.

A streaming ``message_update`` carries the FULL cumulative assistant message (or growing
tool-call ``arguments``) in ``assistantMessageEvent.partial`` on EVERY token delta; logging
that verbatim grew the live log O(n^2) (a ~37 KB file streamed over ~11k deltas produced a
180 MB log). ``_make_live_log_safe_chunk`` drops that redundant cumulative snapshot from the
ON-DISK copy while keeping the incremental ``delta`` and never mutating the original chunk.
"""

import copy
import json

from claude_code_cli_runner.runner import _make_live_log_safe_chunk


def _streaming_tool_call_update(cumulative_arguments_content: str) -> dict:
    return {
        "type": "message_update",
        "assistantMessageEvent": {
            "type": "toolcall_delta",
            "contentIndex": 0,
            "delta": " time",
            "partial": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "abc", "name": "write",
                     "arguments": {"content": cumulative_arguments_content}},
                ],
            },
        },
    }


def test_cumulative_partial_is_dropped_but_delta_is_kept():
    big = "x" * 100_000
    chunk = _streaming_tool_call_update(big)
    trimmed = _make_live_log_safe_chunk(chunk)
    event = trimmed["assistantMessageEvent"]
    assert "partial" not in event  # the O(n^2) cumulative snapshot is gone
    assert event["delta"] == " time"  # the incremental delta is preserved
    assert event["type"] == "toolcall_delta"
    # The trimmed on-disk line is tiny even though the cumulative content was huge.
    assert len(json.dumps(trimmed)) < 500


def test_original_chunk_is_not_mutated():
    chunk = _streaming_tool_call_update("y" * 5000)
    before = copy.deepcopy(chunk)
    _make_live_log_safe_chunk(chunk)
    assert chunk == before  # in-memory consumers still see the full chunk


def test_message_update_without_partial_is_unchanged():
    chunk = {"type": "message_update",
             "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}}
    assert _make_live_log_safe_chunk(chunk) == chunk


def test_non_message_update_chunks_pass_through_untouched():
    for chunk in (
        {"type": "session", "session_id": "s-1"},
        {"type": "result", "result": "done"},
        {"type": "tool_execution_end", "toolName": "bash"},
        "not a dict",
    ):
        assert _make_live_log_safe_chunk(chunk) == chunk


if __name__ == "__main__":
    import unittest

    unittest.main()

"""The opencode_cli harness: argv building, request validation, event
translation, and a full streamed run against the stub opencode."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.opencode_event_translation import (
    OpencodeEventToClaudeChunkTranslator,
    map_opencode_token_block_to_usage,
)
from claude_code_cli_runner.request import (
    HARNESS_OPENCODE_CLI,
    ImageBlock,
)
from claude_code_cli_runner.transports import (
    build_base_harness_argv,
    build_base_opencode_argv,
)

STUB_OPENCODE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stub_streaming_opencode.py"
)


def stub_opencode_build_command(run_request):
    return [sys.executable, STUB_OPENCODE_PATH]


def _opencode_request(tmp_path, **overrides):
    fields = dict(
        input_content=[TextBlock(text="hello opencode stub")],
        workspace_directory=str(tmp_path),
        harness=HARNESS_OPENCODE_CLI,
    )
    fields.update(overrides)
    return RunRequest(**fields)


# --- argv building -----------------------------------------------------------

def test_opencode_argv_carries_run_json_model_and_auto(tmp_path):
    request = _opencode_request(
        tmp_path,
        model="ollama/gemma4:e4b",
        dangerously_skip_permissions=True,
    )
    argv = build_base_opencode_argv(request)
    assert argv[0] == "opencode"
    assert argv[1] == "run"
    assert "--format" in argv and "json" in argv
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "ollama/gemma4:e4b"
    assert "--auto" in argv
    # reasoning/thinking blocks must reach the event stream (some local models
    # answer ONLY in reasoning).
    assert "--thinking" in argv
    # the prompt is NEVER on argv
    assert "hello opencode stub" not in " ".join(argv)


def test_opencode_argv_resumes_existing_session(tmp_path):
    request = _opencode_request(
        tmp_path, session_id="ses_existing_123", resume_session=True
    )
    argv = build_base_opencode_argv(request)
    assert "--session" in argv
    assert argv[argv.index("--session") + 1] == "ses_existing_123"


def test_harness_argv_dispatch_selects_builder(tmp_path):
    opencode_argv = build_base_harness_argv(_opencode_request(tmp_path))
    assert opencode_argv[:2] == ["opencode", "run"]
    claude_argv = build_base_harness_argv(
        RunRequest(
            input_content=[TextBlock(text="x")],
            workspace_directory=str(tmp_path),
        )
    )
    assert claude_argv[0] == "claude"


# --- request validation ------------------------------------------------------

def test_unknown_harness_rejected(tmp_path):
    with pytest.raises(ValueError):
        RunRequest(
            input_content=[TextBlock(text="x")],
            workspace_directory=str(tmp_path),
            harness="who_knows_cli",
        )


def test_opencode_rejects_permission_mode(tmp_path):
    with pytest.raises(ValueError):
        _opencode_request(tmp_path, permission_mode="acceptEdits")


def test_opencode_rejects_caller_chosen_create_session_id(tmp_path):
    with pytest.raises(ValueError):
        _opencode_request(tmp_path, session_id="ses_new", resume_session=False)


def test_opencode_rejects_non_text_content(tmp_path):
    with pytest.raises(ValueError):
        _opencode_request(
            tmp_path,
            input_content=[ImageBlock(mime_type="image/png", data_base64="aaaa")],
        )


# --- event translation -------------------------------------------------------

def test_token_block_mapping():
    usage = map_opencode_token_block_to_usage(
        {"input": 100, "output": 10, "cache": {"write": 7, "read": 3}}
    )
    assert usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 3,
    }


def test_translator_final_stop_step_emits_usage_then_result():
    translator = OpencodeEventToClaudeChunkTranslator()
    translator.translate(
        {"type": "text", "sessionID": "ses_a", "part": {"text": "hello "}}
    )
    translator.translate(
        {"type": "text", "sessionID": "ses_a", "part": {"text": "world"}}
    )
    chunks = translator.translate(
        {
            "type": "step_finish",
            "sessionID": "ses_a",
            "part": {"reason": "stop", "tokens": {"input": 5, "output": 2}},
        }
    )
    assert [chunk["type"] for chunk in chunks] == ["stream_event", "result"]
    assert chunks[1]["result"] == "hello world"
    assert chunks[1]["session_id"] == "ses_a"


def test_translator_intermediate_step_emits_usage_only():
    translator = OpencodeEventToClaudeChunkTranslator()
    chunks = translator.translate(
        {
            "type": "step_finish",
            "part": {"reason": "tool-calls", "tokens": {"input": 5, "output": 2}},
        }
    )
    assert [chunk["type"] for chunk in chunks] == ["stream_event"]


def test_translator_uses_reasoning_text_when_run_ends_with_no_final_text():
    # Observed with ollama gemma4:e4b: the whole answer lands in a reasoning
    # part and the step stops with NO text part. The reasoning text is then the
    # only reply there is — it becomes the (flagged) result.
    translator = OpencodeEventToClaudeChunkTranslator()
    translator.translate(
        {"type": "reasoning", "sessionID": "ses_r",
         "part": {"type": "reasoning", "text": "the actual answer lives here"}}
    )
    chunks = translator.translate(
        {"type": "step_finish", "sessionID": "ses_r",
         "part": {"reason": "stop", "tokens": {"input": 5, "output": 2}}}
    )
    result_chunk = chunks[-1]
    assert result_chunk["type"] == "result"
    assert result_chunk["result"] == "the actual answer lives here"
    assert result_chunk["result_text_source"] == "reasoning_only"


def test_translator_prefers_final_text_over_reasoning():
    translator = OpencodeEventToClaudeChunkTranslator()
    translator.translate(
        {"type": "reasoning", "part": {"type": "reasoning", "text": "working..."}}
    )
    translator.translate({"type": "text", "part": {"text": "final reply"}})
    chunks = translator.translate(
        {"type": "step_finish", "part": {"reason": "stop", "tokens": {}}}
    )
    assert chunks[-1]["result"] == "final reply"
    assert "result_text_source" not in chunks[-1]


def test_translator_passes_unknown_events_through():
    translator = OpencodeEventToClaudeChunkTranslator()
    event = {"type": "step_start", "part": {}}
    assert translator.translate(event) == [event]


# --- full streamed run against the stub opencode ------------------------------

def test_opencode_run_streams_text_result_and_token_usage(tmp_path):
    request = _opencode_request(tmp_path)
    result = run_claude_code_task(request, build_command=stub_opencode_build_command)

    assert result.assistant_text == "stub saw: hello opencode stub"
    assert result.final_result_event is not None
    assert result.final_result_event.get("type") == "result"
    assert result.token_usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 3,
        "counted_tokens": 117,
    }
    assert os.path.isfile(result.live_log_path)

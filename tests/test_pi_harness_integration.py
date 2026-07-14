"""The Pi harness integration: capability declaration, launch-argv construction
on Pi's own terms, and Pi-event -> internal-chunk normalization."""

from __future__ import annotations

import os

from claude_code_cli_runner.harness_integration import get_harness_integration
from claude_code_cli_runner.pi_harness import (
    HARNESS_ID_PI,
    PiOutputEventNormalizer,
    split_model_string_into_provider_and_model,
)
from claude_code_cli_runner.request import RunRequest, TextBlock


def _pi_run_request(**overrides) -> RunRequest:
    base = dict(
        input_content=[TextBlock("do the thing")],
        workspace_directory="/tmp/pi-harness-test",
        model="ollama/gemma4-31b-jang-q3:latest",
        harness=HARNESS_ID_PI,
    )
    base.update(overrides)
    return RunRequest(**base)


def test_pi_is_registered_with_reduced_capabilities():
    caps = get_harness_integration(HARNESS_ID_PI).capabilities
    assert caps.supports_operator_permission_mode is False
    assert caps.supports_caller_chosen_session_id is False
    assert caps.supports_session_prime_and_fork is False
    assert caps.supports_multimodal_input is False
    assert caps.supports_mid_run_command_injection is False


def test_model_string_split_into_provider_and_model():
    assert split_model_string_into_provider_and_model("ollama/gemma4:latest", "x") == (
        "ollama",
        "gemma4:latest",
    )
    assert split_model_string_into_provider_and_model("bare-id", "fallbackprov") == (
        "fallbackprov",
        "bare-id",
    )


def test_launch_command_shape_fresh_run(monkeypatch):
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    monkeypatch.delenv("PI_EXTENSION_PATHS", raising=False)
    monkeypatch.delenv("PI_COMMAND", raising=False)
    argv = get_harness_integration(HARNESS_ID_PI).build_launch_command(_pi_run_request())
    assert argv[0] == "pi"
    assert "--print" in argv and "--mode" in argv
    # provider/model come from the "ollama/..." model string
    assert argv[argv.index("--provider") + 1] == "ollama"
    assert argv[argv.index("--model") + 1] == "gemma4-31b-jang-q3:latest"
    # fresh run: no --session
    assert "--session" not in argv
    for disabling_flag in (
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-extensions",
    ):
        assert disabling_flag in argv


def test_launch_command_resume_adds_session(monkeypatch):
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    argv = get_harness_integration(HARNESS_ID_PI).build_launch_command(
        _pi_run_request(session_id="abc-123", resume_session=True)
    )
    assert argv[argv.index("--session") + 1] == "abc-123"


def test_launch_command_extension_paths_and_tools(monkeypatch):
    monkeypatch.setenv("PI_EXTENSION_PATHS", "/x/web_search.ts")
    argv = get_harness_integration(HARNESS_ID_PI).build_launch_command(
        _pi_run_request(
            assigned_tool_list=["bash", "read", "web_search"],
            restrict_to_assigned_tools_as_whitelist=True,
        )
    )
    assert argv[argv.index("-e") + 1] == "/x/web_search.ts"
    assert argv[argv.index("--tools") + 1] == "bash,read,web_search"


def test_launch_command_empty_tool_whitelist_disables_all_tools(monkeypatch):
    monkeypatch.delenv("PI_EXTENSION_PATHS", raising=False)
    argv = get_harness_integration(HARNESS_ID_PI).build_launch_command(
        _pi_run_request(assigned_tool_list=[], restrict_to_assigned_tools_as_whitelist=True)
    )
    assert argv[argv.index("--tools") + 1] == ""


def test_normalizer_maps_pi_events_to_internal_chunks():
    normalizer = PiOutputEventNormalizer()
    assert normalizer.normalize({"type": "session", "id": "sess-9"}) == []
    assert normalizer.normalize({"type": "agent_start"}) == [{"type": "agent_start"}]

    turn_chunks = normalizer.normalize(
        {
            "type": "turn_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "the answer"}],
                "usage": {"input": 100, "output": 7, "cacheRead": 3, "cacheWrite": 2},
            },
        }
    )
    assistant_chunk = turn_chunks[0]
    assert assistant_chunk["type"] == "assistant"
    assert assistant_chunk["message"]["content"][0]["text"] == "the answer"
    delta_chunk = turn_chunks[1]
    assert delta_chunk["type"] == "stream_event"
    assert delta_chunk["event"]["type"] == "message_delta"
    assert delta_chunk["event"]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 7,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 3,
    }

    result_chunks = normalizer.normalize({"type": "agent_end", "messages": []})
    result = result_chunks[0]
    assert result["type"] == "result"
    assert result["result"] == "the answer"
    assert result["session_id"] == "sess-9"
    assert result["harness"] == "pi"
    assert result["usage"]["input_tokens"] == 100

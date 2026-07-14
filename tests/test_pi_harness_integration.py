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


def test_pi_ignores_permission_mode_instead_of_raising():
    # A collaborative task carries a permission_mode; pi has no operator permission
    # protocol but is always full-auto, so it must SILENTLY DROP the posture (not raise)
    # and still build a valid request.
    run_request = _pi_run_request(permission_mode="acceptEdits")
    assert run_request.permission_mode is None


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
    # streaming form: --mode json WITHOUT --print (which would buffer)
    assert "--print" not in argv
    assert argv[argv.index("--mode") + 1] == "json"
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


class _FakeStdin:
    def __init__(self):
        self.written = ""
        self.closed = False

    def write(self, text):
        self.written += text

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True
        return 0


def test_pi_followup_turn_launches_fresh_resume_process(monkeypatch):
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    monkeypatch.delenv("PI_EXTENSION_PATHS", raising=False)
    integration = get_harness_integration(HARNESS_ID_PI)
    exited_process = _FakeProcess()
    launched = {}

    def fake_launch_harness_subprocess(argv):
        launched["argv"] = argv
        return _FakeProcess()

    followup_process = integration.deliver_followup_turn(
        current_process=exited_process,
        message_text="<tool result here>",
        session_id="sess-abc",
        run_request=_pi_run_request(),
        launch_harness_subprocess=fake_launch_harness_subprocess,
    )

    # The exited per-turn process is awaited, and a NEW resume process is launched.
    assert exited_process.waited is True
    assert followup_process is not exited_process
    argv = launched["argv"]
    assert argv[argv.index("--session") + 1] == "sess-abc"
    # The tool result is delivered on the new process's stdin, then closed.
    assert followup_process.stdin.written == "<tool result here>"
    assert followup_process.stdin.closed is True


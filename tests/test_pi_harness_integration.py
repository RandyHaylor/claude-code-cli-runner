"""The Pi harness integration: capability declaration, launch-argv construction
on Pi's own terms, and Pi-event -> internal-chunk normalization."""

from __future__ import annotations

import json
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
    # RPC mode keeps stdin open, so mid-run command injection IS supported now.
    assert caps.supports_mid_run_command_injection is True


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
    # persistent bidirectional form: --mode rpc WITHOUT --print (which would buffer)
    assert "--print" not in argv
    assert argv[argv.index("--mode") + 1] == "rpc"
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


def test_normalizer_skips_rpc_command_response_objects():
    # RPC command acknowledgements are protocol bookkeeping, not agent events.
    normalizer = PiOutputEventNormalizer()
    assert normalizer.normalize(
        {"type": "response", "command": "prompt", "success": True}
    ) == []
    assert normalizer.normalize(
        {"type": "response", "command": "abort", "success": True}
    ) == []


def test_normalizer_surfaces_session_id_from_get_state_response():
    # RPC mode's ONLY session-id announcement is get_state's response; the
    # normalizer surfaces it as the uniform session chunk so the core captures
    # it — including for runs aborted before agent_end.
    normalizer = PiOutputEventNormalizer()
    chunks = normalizer.normalize(
        {
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"sessionId": "sess-from-get-state", "sessionFile": "/x.jsonl"},
        }
    )
    assert chunks == [{"type": "session", "session_id": "sess-from-get-state"}]
    result_chunks = normalizer.normalize({"type": "agent_end", "messages": []})
    assert result_chunks[0]["session_id"] == "sess-from-get-state"


def test_normalizer_maps_pi_events_to_internal_chunks():
    normalizer = PiOutputEventNormalizer()
    # The session announcement is emitted in the UNIFORM chunk shape (id under
    # "session_id") so the core captures it even on runs aborted before
    # agent_end — same contract as claude's system-init event.
    assert normalizer.normalize({"type": "session", "id": "sess-9"}) == [
        {"type": "session", "session_id": "sess-9"}
    ]
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


def _command_lines(fake_stdin):
    """Parse the LF-delimited JSON command lines written to a fake stdin."""
    return [json.loads(line) for line in fake_stdin.written.splitlines() if line.strip()]


def test_normalizer_reports_aborted_partial_turn_tokens_as_delta_from_baseline():
    # Paused (aborted) runs never reach turn_end — the abort-time
    # get_session_stats read is their ONLY usage report. It must report the
    # DELTA from the baseline (a resumed session starts non-zero), advanced by
    # each completed turn's own usage so nothing is double-counted.
    normalizer = PiOutputEventNormalizer()
    # Baseline read at prompt time: resumed session already holds 1000/200.
    assert normalizer.normalize(
        {"type": "response", "command": "get_session_stats", "success": True,
         "data": {"tokens": {"input": 1000, "output": 200, "cacheRead": 0, "cacheWrite": 0}}}
    ) == []
    assert normalizer.final_usage_after_abort_reported is False
    # A COMPLETED turn reports its own usage (advances the baseline too).
    normalizer.normalize(
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "turn one"}],
         "usage": {"input": 100, "output": 50, "cacheRead": 0, "cacheWrite": 0}}}
    )
    # Abort mid-turn-two, then the final stats read arrives.
    assert normalizer.normalize(
        {"type": "response", "command": "abort", "success": True}
    ) == []
    post_abort_chunks = normalizer.normalize(
        {"type": "response", "command": "get_session_stats", "success": True,
         "data": {"tokens": {"input": 1160, "output": 280, "cacheRead": 0, "cacheWrite": 0}}}
    )
    # Only the aborted partial turn's tokens: 1160-1100=60 in, 280-250=30 out.
    assert post_abort_chunks == [{
        "type": "stream_event",
        "event": {"type": "message_delta", "usage": {
            "input_tokens": 60, "output_tokens": 30,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }},
    }]
    assert normalizer.final_usage_after_abort_reported is True


def test_normalizer_flips_post_abort_flag_even_without_stats_data():
    # A malformed/absent stats payload must still end the core's bounded
    # post-abort read (the flag flips regardless).
    normalizer = PiOutputEventNormalizer()
    normalizer.normalize({"type": "response", "command": "abort", "success": True})
    assert normalizer.normalize(
        {"type": "response", "command": "get_session_stats", "success": False}
    ) == []
    assert normalizer.final_usage_after_abort_reported is True


def test_deliver_prompt_sends_rpc_prompt_command_and_leaves_stdin_open():
    integration = get_harness_integration(HARNESS_ID_PI)
    process = _FakeProcess()
    outcome = integration.deliver_prompt(
        process=process,
        input_content=[TextBlock("do the thing")],
        run_request=_pi_run_request(),
        write_stream_json_message=lambda msg: None,
        permission_prompt_enabled=False,
    )
    # stdin STAYS OPEN (RPC) so follow-ups / abort can be injected.
    assert outcome.process_stdin_remains_open is True
    assert process.stdin.closed is False
    commands = _command_lines(process.stdin)
    # The prompt command, then get_state (session id is ONLY in its response),
    # then get_session_stats (the token baseline for abort-time usage deltas).
    assert len(commands) == 3
    assert commands[0]["type"] == "prompt"
    assert commands[0]["message"] == "do the thing"
    assert commands[0].get("id")  # correlation id present
    assert commands[1]["type"] == "get_state"
    assert commands[1].get("id")
    assert commands[2]["type"] == "get_session_stats"


def test_pi_followup_turn_continues_same_process_in_place():
    integration = get_harness_integration(HARNESS_ID_PI)
    live_process = _FakeProcess()
    launched = {"count": 0}

    def fake_launch_harness_subprocess(argv):
        launched["count"] += 1
        return _FakeProcess()

    followup_process = integration.deliver_followup_turn(
        current_process=live_process,
        message_text="<tool result here>",
        session_id="sess-abc",
        run_request=_pi_run_request(),
        launch_harness_subprocess=fake_launch_harness_subprocess,
    )

    # RPC: SAME process reused (no fresh resume spawn), stdin left OPEN.
    assert followup_process is live_process
    assert launched["count"] == 0
    assert live_process.waited is False
    assert live_process.stdin.closed is False
    commands = _command_lines(live_process.stdin)
    assert commands[-1]["type"] == "prompt"
    assert commands[-1]["message"] == "<tool result here>"


def test_request_abort_sends_rpc_abort_command():
    integration = get_harness_integration(HARNESS_ID_PI)
    process = _FakeProcess()
    integration.request_abort(process=process, write_stream_json_message=lambda msg: None)
    commands = _command_lines(process.stdin)
    # abort, then the final token-stats read for the aborted partial turn.
    assert commands == [{"type": "abort"}, {"type": "get_session_stats"}]


def test_deliver_injected_command_sends_rpc_steer_command():
    # A generic operator "send command" is translated by the pi adapter into an RPC
    # steer command (the core stays harness-agnostic and only passes command_text).
    integration = get_harness_integration(HARNESS_ID_PI)
    process = _FakeProcess()
    integration.deliver_injected_command(
        process=process,
        command_text="focus on the login bug",
        write_stream_json_message=lambda msg: None,
    )
    commands = _command_lines(process.stdin)
    assert commands == [{"type": "steer", "message": "focus on the login bug"}]


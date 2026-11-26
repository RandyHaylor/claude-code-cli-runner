"""Continue the same task in a fresh session from a handoff document
(docs/SESSION-HANDOFF-INTERFACE.md):

- the generic request is detected in turn text and validated;
- the claude argv carries the context-threshold attention hook in its ONE inline
  --settings argument, only when a per-run threshold is set and the gate script exists;
- the hook allows below the threshold, denies past it with the handoff procedure, and
  lets through only a Write to this session's handoff document path;
- end to end against a stub claude: an accepted request starts a NEW claude session with
  --session-id, gives it the handoff path, and the run reports the new session id; a
  refused request is answered inside the same session.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.claude_harness import build_base_claude_argv
from claude_code_cli_runner import fresh_session_from_handoff_document as handoff
from claude_code_cli_runner.http_server import request_from_json

TESTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
HANDOFF_STUB_PATH = os.path.join(
    TESTS_DIRECTORY, "stub_claude_that_requests_a_fresh_session_handoff.py"
)
HOOK_SCRIPT_PATH = os.path.join(
    os.path.dirname(TESTS_DIRECTORY),
    "claude_code_cli_runner",
    "context_threshold_handoff_attention_hook.py",
)
REQUEST_NAME = "continue_this_task_in_a_fresh_session_from_a_handoff_document"


@pytest.fixture
def fake_gate_script(tmp_path, monkeypatch):
    """A stand-in for the claude-code-token-count gate exposing read_last_usage with the
    same contract (last usage's all_observed_tokens)."""
    gate_path = tmp_path / "token_usage_gate.py"
    gate_path.write_text(
        "import json\n"
        "def read_last_usage(transcript_path):\n"
        "    last = {}\n"
        "    for line in open(transcript_path):\n"
        "        usage = (json.loads(line).get('message') or {}).get('usage')\n"
        "        if usage: last = usage\n"
        "    total = sum(int(last.get(k, 0)) for k in ('input_tokens','output_tokens',"
        "'cache_creation_input_tokens','cache_read_input_tokens'))\n"
        "    return {'all_observed_tokens': total}\n"
    )
    monkeypatch.setenv(handoff.GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE, str(gate_path))
    return str(gate_path)


# ---- request detection / validation -------------------------------------------------

def test_detects_the_generic_request_in_turn_text():
    text = 'x\n```runner-request\n{"request": "%s", "handoff_document_path": "/a/b.md"}\n```' % REQUEST_NAME
    assert handoff.detect_fresh_session_handoff_requests_in_turn_text(text) == [
        {"request": REQUEST_NAME, "handoff_document_path": "/a/b.md"}
    ]
    assert handoff.detect_fresh_session_handoff_requests_in_turn_text("no block") == []
    assert handoff.detect_fresh_session_handoff_requests_in_turn_text(
        "```runner-request\nnot json\n```"
    ) == []


def test_validation_refuses_missing_or_empty_documents_and_unknown_requests(tmp_path):
    missing = str(tmp_path / "missing.md")
    empty = tmp_path / "empty.md"
    empty.write_text("  ")
    good = tmp_path / "good.md"
    good.write_text("handoff")
    assert "does not exist" in handoff.reason_the_handoff_request_is_refused(
        {"request": REQUEST_NAME, "handoff_document_path": missing}, str(tmp_path))
    assert "empty" in handoff.reason_the_handoff_request_is_refused(
        {"request": REQUEST_NAME, "handoff_document_path": str(empty)}, str(tmp_path))
    assert "unknown request" in handoff.reason_the_handoff_request_is_refused(
        {"request": "something_else", "handoff_document_path": str(good)}, str(tmp_path))
    assert handoff.reason_the_handoff_request_is_refused(
        {"request": REQUEST_NAME, "handoff_document_path": str(good)}, str(tmp_path)) is None
    # A relative path resolves against the run's working directory.
    assert handoff.reason_the_handoff_request_is_refused(
        {"request": REQUEST_NAME, "handoff_document_path": "good.md"}, str(tmp_path)) is None


# ---- argv: the hook rides the ONE inline --settings argument --------------------------

def _settings_from_argv(argv):
    return json.loads(argv[argv.index("--settings") + 1])


def test_no_threshold_means_no_hook(tmp_path, fake_gate_script):
    argv = build_base_claude_argv(RunRequest(input_content=[TextBlock("x")],
                                             workspace_directory=str(tmp_path)))
    assert "--settings" not in argv


def test_threshold_adds_the_hook_merged_with_caller_settings(tmp_path, fake_gate_script):
    argv = build_base_claude_argv(RunRequest(
        input_content=[TextBlock("x")], workspace_directory=str(tmp_path),
        claude_settings_overrides={"effortLevel": "low",
                                   "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}},
        fresh_session_handoff_context_token_threshold=1234,
    ))
    assert argv.count("--settings") == 1
    settings = _settings_from_argv(argv)
    assert settings["effortLevel"] == "low"
    pre_tool_use = settings["hooks"]["PreToolUse"]
    assert pre_tool_use[0]["matcher"] == "Bash"  # the caller's hook is kept
    command = pre_tool_use[1]["hooks"][0]["command"]
    assert "context_threshold_handoff_attention_hook.py" in command
    assert "--threshold 1234" in command
    assert os.path.join(str(tmp_path), "session_handoff_documents") in command


def test_missing_gate_script_means_no_hook_and_a_stated_reason(tmp_path, monkeypatch):
    monkeypatch.setenv(handoff.GATE_SCRIPT_PATH_ENVIRONMENT_VARIABLE, str(tmp_path / "nope.py"))
    run_request = RunRequest(input_content=[TextBlock("x")], workspace_directory=str(tmp_path),
                             fresh_session_handoff_context_token_threshold=10)
    assert "--settings" not in build_base_claude_argv(run_request)
    assert "not found" in handoff.reason_the_attention_hook_cannot_be_installed(run_request)


def test_remote_execution_means_no_hook(tmp_path, fake_gate_script):
    from claude_code_cli_runner.request import SshConfig
    run_request = RunRequest(input_content=[TextBlock("x")], workspace_directory=str(tmp_path),
                             execution_location="vm_over_ssh", ssh=SshConfig(host="h"),
                             fresh_session_handoff_context_token_threshold=10)
    assert "--settings" not in build_base_claude_argv(run_request)
    assert handoff.reason_the_attention_hook_cannot_be_installed(run_request)


# ---- the hook itself -----------------------------------------------------------------

def _run_hook(tmp_path, tokens, tool_name, tool_input, threshold=100, first_turn_tokens=10):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"message": {"usage": {"input_tokens": first_turn_tokens}}}) + "\n"
        + json.dumps({"message": {"usage": {"input_tokens": tokens}}}) + "\n")
    payload = {"session_id": "s1", "transcript_path": str(transcript), "cwd": str(tmp_path),
               "tool_name": tool_name, "tool_input": tool_input}
    completed = subprocess.run(
        [sys.executable, HOOK_SCRIPT_PATH, "--threshold", str(threshold),
         "--handoff-document-directory", str(tmp_path / "session_handoff_documents")],
        input=json.dumps(payload), capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout) if completed.stdout.strip() else None


def test_hook_allows_below_threshold(tmp_path, fake_gate_script):
    assert _run_hook(tmp_path, 50, "Bash", {"command": "ls"}) is None


def test_hook_denies_past_threshold_with_the_handoff_procedure(tmp_path, fake_gate_script):
    decision = _run_hook(tmp_path, 150, "Bash", {"command": "ls"})
    specific = decision["hookSpecificOutput"]
    assert specific["permissionDecision"] == "deny"
    expected_path = str(tmp_path / "session_handoff_documents" / "handoff_from_session_s1.md")
    assert expected_path in specific["permissionDecisionReason"]
    assert "runner-request" in specific["permissionDecisionReason"]
    assert REQUEST_NAME in specific["permissionDecisionReason"]


def test_hook_skips_handoff_when_the_session_started_over_the_threshold(tmp_path, fake_gate_script):
    # Seen live 2026-11-25: a threshold below the starting context made every fresh
    # session hand off on its first tool call, forever. Such a session is let through.
    decision = _run_hook(tmp_path, 150, "Bash", {"command": "ls"}, first_turn_tokens=120)
    assert decision is None or "hookSpecificOutput" not in decision
    assert "would loop" in (decision or {}).get("systemMessage", "")


def test_hook_lets_through_only_the_write_of_the_handoff_document(tmp_path, fake_gate_script):
    handoff_path = str(tmp_path / "session_handoff_documents" / "handoff_from_session_s1.md")
    assert _run_hook(tmp_path, 150, "Write", {"file_path": handoff_path, "content": "x"}) is None
    other = _run_hook(tmp_path, 150, "Write", {"file_path": str(tmp_path / "x.py"), "content": "x"})
    assert other["hookSpecificOutput"]["permissionDecision"] == "deny"


# ---- end to end against the stub claude ----------------------------------------------

def _run_stub(tmp_path, monkeypatch, write_document):
    os.chmod(HANDOFF_STUB_PATH, 0o755)
    document_path = str(tmp_path / "work" / "handoff.md")
    os.makedirs(os.path.dirname(document_path), exist_ok=True)
    monkeypatch.setenv("STUB_HANDOFF_DOCUMENT_PATH", document_path)
    monkeypatch.setenv("STUB_WRITE_HANDOFF_DOCUMENT", "1" if write_document else "0")
    run_request = RunRequest(
        input_content=[TextBlock("do the task")],
        workspace_directory=str(tmp_path / "work"),
        claude_command=HANDOFF_STUB_PATH,
        timeout_seconds=30,
    )
    return run_claude_code_task(run_request, build_command=build_base_claude_argv), document_path


def test_accepted_request_starts_a_fresh_session_that_gets_the_handoff_path(tmp_path, monkeypatch):
    result, document_path = _run_stub(tmp_path, monkeypatch, write_document=True)
    assert "FRESH_SESSION_GOT::" in result.assistant_text
    assert document_path in result.assistant_text
    records = result.fresh_sessions_started_from_handoff_documents
    assert len(records) == 1
    assert records[0]["previous_session_id"] == "stub-first-session"
    assert records[0]["handoff_document_path"] == document_path
    new_session_id = records[0]["new_session_id"]
    assert new_session_id and new_session_id != "stub-first-session"
    # The run reports the NEW session (the existing path Unharness records on the task).
    assert result.harness_session_id == new_session_id
    assert result.final_result_event["session_id"] == new_session_id
    assert result.to_dict()["fresh_sessions_started_from_handoff_documents"] == records
    log_text = open(result.live_log_path, encoding="utf-8").read()
    assert "started a fresh session" in log_text


def test_refused_request_is_answered_inside_the_same_session(tmp_path, monkeypatch):
    result, _ = _run_stub(tmp_path, monkeypatch, write_document=False)
    assert "FIRST_SESSION_GOT::" in result.assistant_text
    assert "does not exist" in result.assistant_text
    assert result.fresh_sessions_started_from_handoff_documents == []
    assert result.harness_session_id == "stub-first-session"


# ---- the HTTP wire carries the per-run threshold -------------------------------------

def test_http_wire_carries_the_threshold(tmp_path):
    run_request = request_from_json({
        "input_content": [{"block_type": "text", "text": "x"}],
        "workspace_directory": str(tmp_path),
        "fresh_session_handoff_context_token_threshold": 5000,
    })
    assert run_request.fresh_session_handoff_context_token_threshold == 5000

"""raw-1216/1222: the resume-failure fallback.

A claude RESUME run whose session transcript is GONE from the task cwd cannot be
resumed. When the request carries a ``resume_fallback_prompt`` (the full prompt,
built fresh per dispatch, never persisted), the runner detects the missing
transcript deterministically (a file pre-check, no stderr parsing) and runs a
FRESH session with that prompt instead of failing. claude is always the stub.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner import claude_session_store
from claude_code_cli_runner import runner as runner_mod
from conftest import STUB_AS_CLAUDE

DEAD_SESSION_ID = "11111111-2222-3333-4444-555555555555"
FALLBACK_PROMPT_TEXT = "FULL FALLBACK PROMPT: context + task + reply format"
INCREMENTAL_TURN_TEXT = "the incremental resume turn"


def _capture_popen(monkeypatch):
    """Record every argv passed to subprocess.Popen in the runner, delegating to
    the real Popen so the stub still runs."""
    captured = []
    real_popen = subprocess.Popen

    def recording_popen(argv, *args, **kwargs):
        captured.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(runner_mod.subprocess, "Popen", recording_popen)
    return captured


def _read_log(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _resume_request(tmp_path, fallback_prompt):
    return RunRequest(
        input_content=[TextBlock(text=INCREMENTAL_TURN_TEXT)],
        workspace_directory=str(tmp_path / "ws"),
        claude_command=STUB_AS_CLAUDE,
        session_id=DEAD_SESSION_ID,
        resume_session=True,
        resume_fallback_prompt=fallback_prompt,
    )


def test_missing_transcript_runs_fresh_session_with_fallback_prompt(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", str(tmp_path / "projects"))
    captured = _capture_popen(monkeypatch)

    result = run_claude_code_task(_resume_request(tmp_path, FALLBACK_PROMPT_TEXT))

    # One FRESH run: no --resume, no --session-id (claude mints the new id).
    assert len(captured) == 1
    fresh_argv = captured[0]
    assert "--resume" not in fresh_argv
    assert "--session-id" not in fresh_argv

    # The input carries the FULL fallback prompt, not the incremental turn.
    log = _read_log(result.live_log_path)
    assert FALLBACK_PROMPT_TEXT in log
    assert INCREMENTAL_TURN_TEXT not in log
    # The fallback is announced as a startup note in the live log.
    assert "runner_note" in log
    assert "resume of session %s impossible" % DEAD_SESSION_ID in log
    assert result.final_result_event is not None


def test_present_transcript_resumes_normally(tmp_path, monkeypatch):
    projects_root = str(tmp_path / "projects")
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", projects_root)
    transcript_path = claude_session_store.session_jsonl_path(
        str(tmp_path / "ws"), DEAD_SESSION_ID, projects_root=projects_root
    )
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    captured = _capture_popen(monkeypatch)

    result = run_claude_code_task(_resume_request(tmp_path, FALLBACK_PROMPT_TEXT))

    # Normal resume: --resume <sid>, and the incremental turn is sent.
    assert len(captured) == 1
    resume_argv = captured[0]
    assert "--resume" in resume_argv
    assert resume_argv[resume_argv.index("--resume") + 1] == DEAD_SESSION_ID
    log = _read_log(result.live_log_path)
    assert INCREMENTAL_TURN_TEXT in log
    assert FALLBACK_PROMPT_TEXT not in log


def test_missing_transcript_without_fallback_prompt_keeps_old_behavior(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", str(tmp_path / "projects"))
    captured = _capture_popen(monkeypatch)

    run_claude_code_task(_resume_request(tmp_path, None))

    # No fallback prompt carried -> the runner attempts the resume as before.
    assert len(captured) == 1
    assert "--resume" in captured[0]

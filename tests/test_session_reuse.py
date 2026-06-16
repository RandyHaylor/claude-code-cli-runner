"""Tests for the OPTIONAL prime-once / fork-per-task session-reuse primitive.

These assert the RUNNER's logic (registry bookkeeping + which argv it builds +
the inline fallback), NOT claude's real session store — claude is always the
stub, which ignores --resume/--fork-session/--session-id and still streams.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import (
    ReusableContext,
    RunRequest,
    TextBlock,
    run_claude_code_task,
)
from claude_code_cli_runner import runner as runner_mod
from claude_code_cli_runner import session_registry
from conftest import STUB_AS_CLAUDE, stub_build_command


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


def test_first_run_primes_and_task_forks(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    captured = _capture_popen(monkeypatch)

    request = RunRequest(
        input_content=[TextBlock(text="the task remainder")],
        workspace_directory=str(tmp_path / "ws"),
        claude_command=STUB_AS_CLAUDE,
        reusable_context=ReusableContext(
            chunk_id="chunkA",
            content=[TextBlock(text="big leading reusable context")],
        ),
    )
    result = run_claude_code_task(request, registry_path=registry)

    # Registry now maps chunkA -> a primed session id.
    primed_sid = session_registry.get_primed_session_id("chunkA", registry_path=registry)
    assert primed_sid and primed_sid.startswith("primed-chunkA-")

    # Two Popens: [0] prime (plain --session-id), [1] task (fork).
    assert len(captured) == 2
    prime_argv, task_argv = captured
    assert "--session-id" in prime_argv
    assert primed_sid in prime_argv
    assert "--resume" not in prime_argv

    # The task run forks from the primed sid with a FRESH task sid.
    assert "--resume" in task_argv
    assert task_argv[task_argv.index("--resume") + 1] == primed_sid
    assert "--fork-session" in task_argv
    assert "--session-id" in task_argv
    fresh_sid = task_argv[task_argv.index("--session-id") + 1]
    assert fresh_sid.startswith("task-chunkA-")

    # The chunk text is NOT re-sent in the task run (it lives in the primed
    # session); the task user message carries only the per-task remainder.
    log = _read_log(result.live_log_path)
    assert "the task remainder" in log
    assert "big leading reusable context" not in log
    assert result.final_result_event is not None


def test_second_run_same_chunk_does_not_reprime(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")

    base = dict(
        workspace_directory=str(tmp_path / "ws"),
        claude_command=STUB_AS_CLAUDE,
    )

    # First run primes.
    run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task one")],
            reusable_context=ReusableContext(
                chunk_id="chunkB", content=[TextBlock(text="ctx")]
            ),
            **base,
        ),
        registry_path=registry,
    )
    primed_sid = session_registry.get_primed_session_id("chunkB", registry_path=registry)

    # Second run, same chunk_id: capture argvs — there must be NO prime, just one
    # fork that resumes the SAME primed sid.
    captured = _capture_popen(monkeypatch)
    run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task two")],
            reusable_context=ReusableContext(
                chunk_id="chunkB", content=[TextBlock(text="ctx")]
            ),
            **base,
        ),
        registry_path=registry,
    )

    assert len(captured) == 1  # only the task fork, no re-prime
    task_argv = captured[0]
    assert "--resume" in task_argv
    assert task_argv[task_argv.index("--resume") + 1] == primed_sid
    # Registry unchanged.
    assert session_registry.get_primed_session_id("chunkB", registry_path=registry) == primed_sid


def test_disable_reuse_prepends_inline_no_registry_write(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    captured = _capture_popen(monkeypatch)

    request = RunRequest(
        input_content=[TextBlock(text="task body")],
        workspace_directory=str(tmp_path / "ws"),
        claude_command="claude",
        enable_session_reuse=False,
        reusable_context=ReusableContext(
            chunk_id="chunkC",
            content=[TextBlock(text="leading chunk inline")],
        ),
    )
    result = run_claude_code_task(
        request, build_command=stub_build_command, registry_path=registry
    )

    # No registry file written / no entry.
    assert session_registry.get_primed_session_id("chunkC", registry_path=registry) is None
    assert not os.path.isfile(registry)

    # Single run, no --resume; chunk PREPENDED inline (chunk text reaches claude).
    assert len(captured) == 1
    assert "--resume" not in captured[0]
    log = _read_log(result.live_log_path)
    assert "leading chunk inline" in log
    assert "task body" in log


def test_reuse_failure_falls_back_inline_with_note(tmp_path):
    registry = str(tmp_path / "reg.json")

    # claude_command points at a nonexistent binary so the PRIME Popen raises;
    # build_command (the stub) is used for the inline fallback and succeeds.
    request = RunRequest(
        input_content=[TextBlock(text="task survives")],
        workspace_directory=str(tmp_path / "ws"),
        claude_command="/nonexistent/claude-binary-xyz",
        reusable_context=ReusableContext(
            chunk_id="chunkD",
            content=[TextBlock(text="fallback leading chunk")],
        ),
    )
    result = run_claude_code_task(
        request, build_command=stub_build_command, registry_path=registry
    )

    # Task still completed via the inline fallback.
    assert result.final_result_event is not None
    # A fallback note was recorded in the live log.
    log = _read_log(result.live_log_path)
    assert "runner_note" in log
    assert "falling back to inline" in log
    # Chunk was prepended inline (so it reached claude after all).
    assert "fallback leading chunk" in log
    assert "task survives" in log
    # The unusable chunk id was forgotten (not left pointing at a bad session).
    assert session_registry.get_primed_session_id("chunkD", registry_path=registry) is None


def test_no_reusable_context_behaves_as_before(tmp_path):
    registry = str(tmp_path / "reg.json")
    request = RunRequest(
        input_content=[TextBlock(text="plain task")],
        workspace_directory=str(tmp_path / "ws"),
    )
    os.environ["STUB_RESULT_TEXT"] = "plain answer"
    try:
        result = run_claude_code_task(
            request, build_command=stub_build_command, registry_path=registry
        )
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)

    assert result.assistant_text == "plain answer"
    assert not os.path.isfile(registry)
    log = _read_log(result.live_log_path)
    assert "plain task" in log
    assert "runner_note" not in log

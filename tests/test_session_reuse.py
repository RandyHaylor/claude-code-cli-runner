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
    ImageBlock,
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
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", str(tmp_path / "projects"))
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
    # claude --session-id requires a valid UUID; the chunk_id<->session link
    # lives in the registry mapping, not in the id string.
    assert primed_sid
    import uuid as _uuid
    _uuid.UUID(primed_sid)  # raises if not a valid UUID

    # Two Popens: [0] prime (SIMPLE completing call), [1] task (fork).
    assert len(captured) == 2
    prime_argv, task_argv = captured
    # The prime is the SIMPLE form: --session-id + -p <chunk text>, and it must
    # NOT use the streaming stream-json flags (it completes on its own).
    assert "--session-id" in prime_argv
    assert primed_sid in prime_argv
    assert "--resume" not in prime_argv
    assert "-p" in prime_argv
    assert prime_argv[prime_argv.index("-p") + 1] == "big leading reusable context"
    assert "--input-format" not in prime_argv
    assert "stream-json" not in prime_argv
    assert "--output-format" not in prime_argv
    assert "--include-partial-messages" not in prime_argv

    # The task run forks from the primed sid with a FRESH task sid.
    assert "--resume" in task_argv
    assert task_argv[task_argv.index("--resume") + 1] == primed_sid
    assert "--fork-session" in task_argv
    assert "--session-id" in task_argv
    fresh_sid = task_argv[task_argv.index("--session-id") + 1]
    _uuid.UUID(fresh_sid)  # the task sid is a fresh valid UUID
    assert fresh_sid != primed_sid

    # The chunk text is NOT re-sent in the task run (it lives in the primed
    # session); the task user message carries only the per-task remainder.
    log = _read_log(result.live_log_path)
    assert "the task remainder" in log
    assert "big leading reusable context" not in log
    assert result.final_result_event is not None


def test_second_run_same_chunk_does_not_reprime(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", str(tmp_path / "projects"))

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


def test_non_text_chunk_falls_back_inline_no_prime(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    captured = _capture_popen(monkeypatch)

    # A chunk with a non-text (image) block can't be primed via the SIMPLE
    # completing -p call -> the runner must NOT prime, and falls back to inline.
    request = RunRequest(
        input_content=[TextBlock(text="task body")],
        workspace_directory=str(tmp_path / "ws"),
        claude_command=STUB_AS_CLAUDE,
        reusable_context=ReusableContext(
            chunk_id="chunkIMG",
            content=[
                TextBlock(text="some caption"),
                ImageBlock(mime_type="image/png", data_base64="aGVsbG8="),
            ],
        ),
    )
    result = run_claude_code_task(
        request, build_command=stub_build_command, registry_path=registry
    )

    # Only one Popen (the inline task run); no prime was attempted.
    assert len(captured) == 1
    assert "-p" not in captured[0] or "--resume" not in captured[0]
    assert "--resume" not in captured[0]
    # Nothing recorded for this chunk; fallback note present.
    assert session_registry.get_primed_session_id("chunkIMG", registry_path=registry) is None
    log = _read_log(result.live_log_path)
    assert "runner_note" in log
    assert "falling back to inline" in log
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


# --- claude_session_store helpers -----------------------------------------

from claude_code_cli_runner import claude_session_store as store


def test_encode_cwd_replaces_slash_and_underscore():
    assert store.encode_cwd_to_project_dirname("/tmp/ccrA") == "-tmp-ccrA"
    assert store.encode_cwd_to_project_dirname("/home/aikenyon/x_y") == "-home-aikenyon-x-y"


def test_project_dir_and_jsonl_path_use_override(tmp_path):
    root = str(tmp_path / "projects")
    pdir = store.project_dir_for_cwd("/tmp/ccrA", projects_root=root)
    assert pdir == os.path.join(root, "-tmp-ccrA")
    jp = store.session_jsonl_path("/tmp/ccrA", "sid123", projects_root=root)
    assert jp == os.path.join(root, "-tmp-ccrA", "sid123.jsonl")


def test_ensure_session_present_copies_into_target_cwd(tmp_path):
    root = str(tmp_path / "projects")
    # source jsonl as if primed in cwd A
    src = store.session_jsonl_path("/tmp/ccrA", "sid", projects_root=root)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "w", encoding="utf-8") as h:
        h.write("primed\n")
    target = store.ensure_session_present_in_cwd("sid", src, "/tmp/ccrB", projects_root=root)
    assert target == store.session_jsonl_path("/tmp/ccrB", "sid", projects_root=root)
    assert os.path.isfile(target)
    with open(target, encoding="utf-8") as h:
        assert h.read() == "primed\n"


def test_ensure_session_present_same_cwd_is_noop(tmp_path):
    root = str(tmp_path / "projects")
    src = store.session_jsonl_path("/tmp/ccrA", "sid", projects_root=root)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "w", encoding="utf-8") as h:
        h.write("x\n")
    target = store.ensure_session_present_in_cwd("sid", src, "/tmp/ccrA", projects_root=root)
    assert os.path.abspath(target) == os.path.abspath(src)


def test_ensure_session_present_missing_source_raises(tmp_path):
    root = str(tmp_path / "projects")
    import pytest
    with pytest.raises(FileNotFoundError):
        store.ensure_session_present_in_cwd(
            "sid", str(tmp_path / "nope.jsonl"), "/tmp/ccrB", projects_root=root
        )


# --- cross-cwd reuse in the runner ----------------------------------------

def test_registry_records_session_and_source_jsonl(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    projects = str(tmp_path / "projects")
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", projects)

    prime_ws = str(tmp_path / "ws_prime")
    run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task one")],
            workspace_directory=prime_ws,
            claude_command=STUB_AS_CLAUDE,
            reusable_context=ReusableContext(
                chunk_id="chunkX", content=[TextBlock(text="ctx")]
            ),
        ),
        registry_path=registry,
    )
    record = session_registry.get_primed_record("chunkX", registry_path=registry)
    assert record is not None
    assert record["session_id"]
    # source jsonl is the prime cwd's project-dir transcript, and it exists.
    expected = store.session_jsonl_path(
        prime_ws, record["session_id"], projects_root=projects
    )
    assert record["source_jsonl"] == expected
    assert os.path.isfile(expected)


def test_fork_in_different_cwd_relocates_jsonl_then_forks(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    projects = str(tmp_path / "projects")
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", projects)

    prime_ws = str(tmp_path / "ws_prime")
    task_ws = str(tmp_path / "ws_task")

    # First run primes in prime_ws.
    run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task one")],
            workspace_directory=prime_ws,
            claude_command=STUB_AS_CLAUDE,
            reusable_context=ReusableContext(
                chunk_id="chunkR", content=[TextBlock(text="ctx")]
            ),
        ),
        registry_path=registry,
    )
    primed_sid = session_registry.get_primed_session_id("chunkR", registry_path=registry)
    target_jsonl = store.session_jsonl_path(task_ws, primed_sid, projects_root=projects)
    assert not os.path.isfile(target_jsonl)  # not yet present in task cwd

    # Second run, SAME chunk, DIFFERENT cwd: must relocate then fork.
    captured = _capture_popen(monkeypatch)
    run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task two")],
            workspace_directory=task_ws,
            claude_command=STUB_AS_CLAUDE,
            reusable_context=ReusableContext(
                chunk_id="chunkR", content=[TextBlock(text="ctx")]
            ),
        ),
        registry_path=registry,
    )
    # The jsonl was copied into the task cwd's project dir BEFORE the fork.
    assert os.path.isfile(target_jsonl)
    # Single Popen (only the fork, no re-prime) and its argv resumes the primed sid.
    assert len(captured) == 1
    task_argv = captured[0]
    assert "--resume" in task_argv
    assert task_argv[task_argv.index("--resume") + 1] == primed_sid
    assert "--fork-session" in task_argv


def test_missing_source_jsonl_falls_back_inline(tmp_path, monkeypatch):
    registry = str(tmp_path / "reg.json")
    projects = str(tmp_path / "projects")
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", projects)

    # Pre-seed the registry with a primed session whose source jsonl does NOT
    # exist -> the fork relocate raises -> inline fallback.
    session_registry.record_primed_session_id(
        "chunkM",
        "11111111-1111-1111-1111-111111111111",
        registry_path=registry,
        source_jsonl=str(tmp_path / "gone.jsonl"),
    )
    captured = _capture_popen(monkeypatch)
    result = run_claude_code_task(
        RunRequest(
            input_content=[TextBlock(text="task body")],
            workspace_directory=str(tmp_path / "ws"),
            claude_command=STUB_AS_CLAUDE,
            reusable_context=ReusableContext(
                chunk_id="chunkM", content=[TextBlock(text="leading chunk inline")]
            ),
        ),
        registry_path=registry,
    )
    assert len(captured) == 1
    assert "--resume" not in captured[0]
    log = _read_log(result.live_log_path)
    assert "falling back to inline" in log
    assert "leading chunk inline" in log
    assert "task body" in log
    # The unusable chunk was forgotten.
    assert session_registry.get_primed_session_id("chunkM", registry_path=registry) is None


def test_registry_backcompat_bare_string_value(tmp_path):
    registry = str(tmp_path / "reg.json")
    with open(registry, "w", encoding="utf-8") as h:
        json.dump({"oldChunk": "abc-123"}, h)
    assert session_registry.get_primed_session_id("oldChunk", registry_path=registry) == "abc-123"
    rec = session_registry.get_primed_record("oldChunk", registry_path=registry)
    assert rec == {"session_id": "abc-123", "source_jsonl": None}

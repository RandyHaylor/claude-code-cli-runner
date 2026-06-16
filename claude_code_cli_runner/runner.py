"""The single unified, always-streaming execution path: run_claude_code_task.

One library API drives every face. It launches the claude streaming process
(argv chosen by the transport for the request's execution_location), delivers
the multimodal prompt over stdin, appends every stream chunk to the live log,
honours the out-of-band control channel (pause/resume/send_command/
end_and_return), reflects the run state to the sidecar, and returns a
multimodal-aware RunResult (assistant text + produced artifacts + final result
event + exit/status + raw stream-log path).

The subprocess command is injectable via ``build_command`` so tests point the
run at a stub claude; the default uses the transport for the request.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

from .content import (
    build_injected_user_message,
    build_user_message,
    list_produced_artifacts,
    snapshot_workspace_files,
)
from .live_files import (
    CONTROL_END_AND_RETURN,
    CONTROL_PAUSE,
    CONTROL_RESUME,
    CONTROL_SEND_COMMAND,
    RUN_STATE_OPERATOR_ENDED,
    RUN_STATE_PAUSED,
    RUN_STATE_RUNNING,
    control_channel_path,
    live_log_path,
    read_new_control_intents,
    run_status_path,
)
from .request import RunRequest
from .result import RunResult
from . import session_registry
from .transports import (
    build_command_for,
    build_fork_claude_argv,
    build_priming_claude_argv,
)


def run_claude_code_task(
    run_request: RunRequest,
    *,
    build_command=None,
    pause_poll_seconds: float = 0.02,
    registry_path=None,
) -> RunResult:
    """Run a streaming claude -p task and return a multimodal RunResult.

    ``build_command(run_request) -> argv`` is injectable (tests pass a stub-
    pointing builder); by default the transport for the request's
    execution_location is used.

    OPTIONAL session reuse: if the request carries a ``reusable_context`` AND
    ``enable_session_reuse`` is True, the leading chunk is primed ONCE (keyed by
    its ``chunk_id`` in the on-disk registry) and the task is run as a FORK of
    that primed session, so the chunk's tokens are cache-reused instead of
    re-sent. This is ALWAYS best-effort: when reuse is disabled, absent, or
    anything fails, the chunk is PREPENDED inline to ``input_content`` and the
    task runs normally (always correct). A note is appended to the live log on
    any fallback.
    """
    if build_command is None:
        build_command = build_command_for

    reuse = run_request.reusable_context
    if reuse is not None and run_request.enable_session_reuse:
        return _run_with_session_reuse(
            run_request,
            build_command=build_command,
            pause_poll_seconds=pause_poll_seconds,
            registry_path=registry_path,
        )

    # No reuse: inline the chunk (if present) and run normally.
    effective_input = _inline_input_content(run_request)
    return _stream_one_run(
        run_request,
        argv=build_command(run_request),
        input_content=effective_input,
        pause_poll_seconds=pause_poll_seconds,
        startup_notes=None,
    )


def _inline_input_content(run_request: RunRequest) -> list:
    """The input content actually sent: when a reusable context is present but
    NOT being reused (disabled / fallback), its blocks are PREPENDED to the
    task's input_content (chunk first). Otherwise just the task input."""
    reuse = run_request.reusable_context
    if reuse is None:
        return list(run_request.input_content)
    return list(reuse.content) + list(run_request.input_content)


def _run_with_session_reuse(
    run_request: RunRequest,
    *,
    build_command,
    pause_poll_seconds: float,
    registry_path,
) -> RunResult:
    """Prime-once / fork-per-task path, with inline fallback on any failure."""
    reuse = run_request.reusable_context
    chunk_id = reuse.chunk_id

    def fall_back_inline(note: str) -> RunResult:
        return _stream_one_run(
            run_request,
            argv=build_command(run_request),
            input_content=_inline_input_content(run_request),
            pause_poll_seconds=pause_poll_seconds,
            startup_notes=[note],
        )

    try:
        primed_sid = session_registry.get_primed_session_id(
            chunk_id, registry_path=registry_path
        )
        notes = []
        if primed_sid is None:
            # PRIME ONCE: run a session that ingests ONLY the chunk, record its id.
            primed_sid = "primed-%s-%s" % (chunk_id, uuid.uuid4().hex[:8])
            prime_argv = build_priming_claude_argv(run_request, primed_sid)
            _run_priming_session(
                run_request, argv=prime_argv, chunk_content=list(reuse.content)
            )
            session_registry.record_primed_session_id(
                chunk_id, primed_sid, registry_path=registry_path
            )
            notes.append(
                "reusable_context %r primed new session %s" % (chunk_id, primed_sid)
            )
        else:
            notes.append(
                "reusable_context %r reusing primed session %s" % (chunk_id, primed_sid)
            )

        # TASK as a FORK of the primed session: send ONLY the per-task remainder
        # (the chunk is already in the primed session, NOT re-sent here).
        task_sid = "task-%s-%s" % (chunk_id, uuid.uuid4().hex[:8])
        fork_argv = build_fork_claude_argv(run_request, primed_sid, task_sid)
        return _stream_one_run(
            run_request,
            argv=fork_argv,
            input_content=list(run_request.input_content),
            pause_poll_seconds=pause_poll_seconds,
            startup_notes=notes,
        )
    except Exception as cause:  # noqa: BLE001 - reuse must NEVER fail the task
        # Best-effort: a stale/unusable session id should not be reused again.
        try:
            session_registry.forget_chunk(chunk_id, registry_path=registry_path)
        except Exception:  # noqa: BLE001
            pass
        return fall_back_inline(
            "session reuse for chunk %r failed (%s); falling back to inline chunk"
            % (chunk_id, cause)
        )


def _run_priming_session(run_request: RunRequest, *, argv, chunk_content) -> None:
    """Run a brief claude session that ingests the chunk, then ends. Not part of
    the live window — its only purpose is to leave a reusable primed session
    behind. Raises on a non-zero/failed invocation so the caller can fall back."""
    workspace_directory = os.fspath(run_request.workspace_directory)
    os.makedirs(workspace_directory, exist_ok=True)
    process = subprocess.Popen(
        argv,
        cwd=workspace_directory,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdin is not None:
            process.stdin.write(json.dumps(build_user_message(chunk_content)) + "\n")
            process.stdin.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass
    # Drain stdout until the session ends (result event) or the process exits.
    saw_result = False
    if process.stdout is not None:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(chunk, dict) and chunk.get("type") == "result":
                saw_result = True
                break
    _terminate_process(process)
    exit_code = process.poll()
    if not saw_result and exit_code not in (0, None):
        raise RuntimeError(
            "priming claude exited %r without a result event" % exit_code
        )


def _stream_one_run(
    run_request: RunRequest,
    *,
    argv,
    input_content,
    pause_poll_seconds: float,
    startup_notes,
) -> RunResult:
    """The single, always-streaming execution core: launch ``argv``, deliver
    ``input_content`` over stdin, honour the control channel, write the live
    window, and return a RunResult. ``startup_notes`` (if any) are appended to
    the live log as ``runner_note`` records before streaming begins."""
    workspace_directory = os.fspath(run_request.workspace_directory)
    os.makedirs(workspace_directory, exist_ok=True)

    log_path = run_request.live_log_path or live_log_path(workspace_directory)
    control_path = (
        run_request.control_channel_path or control_channel_path(workspace_directory)
    )
    status_path = run_request.run_status_path or run_status_path(workspace_directory)

    # Snapshot workspace files BEFORE the run so produced artifacts can be diffed.
    baseline_files = snapshot_workspace_files(workspace_directory)

    # Start clean so a tailer's offsets are meaningful.
    open(log_path, "w", encoding="utf-8").close()
    if startup_notes:
        with open(log_path, "a", encoding="utf-8") as note_handle:
            for note in startup_notes:
                note_handle.write(
                    json.dumps({"received_at": time.time(), "runner_note": note}) + "\n"
                )
            note_handle.flush()
            os.fsync(note_handle.fileno())
    _reflect(status_path, RUN_STATE_RUNNING)

    process = subprocess.Popen(
        argv,
        cwd=workspace_directory,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def write_stream_json_message(message: dict) -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    # Deliver the multimodal prompt as a stdin stream-json user message. stdin
    # stays OPEN afterwards so mid-run send_command injection still works.
    write_stream_json_message(build_user_message(input_content))

    consumed_control_lines = 0
    paused = False
    operator_ended = False
    result_seen = False
    final_result_event = None
    collected_text_parts: list = []
    result_text_fallback: list = []
    deadline = (
        time.monotonic() + run_request.timeout_seconds
        if run_request.timeout_seconds
        else None
    )

    def render_text_from_chunk(chunk):
        nonlocal final_result_event
        if not isinstance(chunk, dict):
            return
        chunk_type = chunk.get("type")
        if chunk_type == "result":
            final_result_event = chunk
            # The result event's text is only a FALLBACK: it usually duplicates
            # the assistant-message text, so we keep it separately and use it
            # only when no assistant text was streamed.
            if isinstance(chunk.get("result"), str):
                result_text_fallback.append(chunk["result"])
        elif chunk_type == "assistant":
            message = chunk.get("message", {})
            blocks = message.get("content", []) if isinstance(message, dict) else []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    collected_text_parts.append(block.get("text", ""))

    log_handle = open(log_path, "a", encoding="utf-8")

    def append_chunk_to_live_log(raw_line: str):
        nonlocal result_seen
        try:
            chunk = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            chunk = None
        record = {"received_at": time.time()}
        if chunk is None:
            record["raw"] = raw_line
        else:
            record["chunk"] = chunk
            render_text_from_chunk(chunk)
            if isinstance(chunk, dict) and chunk.get("type") == "result":
                result_seen = True
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        os.fsync(log_handle.fileno())

    def drain_control_channel() -> bool:
        nonlocal consumed_control_lines, paused, operator_ended
        intents, consumed_control_lines = read_new_control_intents(
            control_path, consumed_control_lines
        )
        for intent in intents:
            kind = intent.get("control_intent")
            if kind == CONTROL_PAUSE:
                paused = True
                _reflect(status_path, RUN_STATE_PAUSED)
            elif kind == CONTROL_RESUME:
                paused = False
                _reflect(status_path, RUN_STATE_RUNNING)
            elif kind == CONTROL_SEND_COMMAND:
                write_stream_json_message(
                    build_injected_user_message(intent.get("command_text", ""))
                )
            elif kind == CONTROL_END_AND_RETURN:
                operator_ended = True
                _reflect(status_path, RUN_STATE_OPERATOR_ENDED)
                return True
        return False

    try:
        for raw_line in process.stdout:
            raw_line = raw_line.rstrip("\n")
            if raw_line == "":
                continue

            if drain_control_channel():
                break

            while paused:
                if drain_control_channel():
                    break
                time.sleep(pause_poll_seconds)
            if operator_ended:
                break

            if deadline is not None and time.monotonic() > deadline:
                break

            append_chunk_to_live_log(raw_line)

            if result_seen:
                break
    finally:
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
        exit_code = None
        if operator_ended or result_seen:
            _terminate_process(process)
        else:
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _terminate_process(process)
        if exit_code is None:
            exit_code = process.poll()

    final_run_state = RUN_STATE_OPERATOR_ENDED if operator_ended else RUN_STATE_RUNNING
    if not operator_ended:
        # Reflect a terminal "running->done" by leaving running; the sidecar's
        # job is the live annotation, and the result carries the real outcome.
        final_run_state = read_run_state_value(status_path) or RUN_STATE_RUNNING

    produced = list_produced_artifacts(workspace_directory, baseline_files)

    assistant_text = "".join(collected_text_parts) or "".join(result_text_fallback)

    return RunResult(
        assistant_text=assistant_text,
        final_result_event=final_result_event,
        produced_artifacts=produced,
        exit_code=exit_code,
        operator_ended=operator_ended,
        live_log_path=log_path,
        run_state=final_run_state,
        workspace_directory=workspace_directory,
    )


def _reflect(status_path: str, run_state: str) -> None:
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump({"run_state": run_state, "updated_at": time.time()}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def read_run_state_value(status_path: str):
    if not os.path.isfile(status_path):
        return None
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        value = parsed.get("run_state")
        if isinstance(value, str):
            return value
    return None


def _terminate_process(process) -> None:
    """Best-effort terminate-then-kill a still-running process and reap it."""
    if process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    except (OSError, ValueError):
        pass

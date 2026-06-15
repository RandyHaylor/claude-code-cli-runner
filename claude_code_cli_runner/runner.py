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
from .transports import build_command_for


def run_claude_code_task(
    run_request: RunRequest,
    *,
    build_command=None,
    pause_poll_seconds: float = 0.02,
) -> RunResult:
    """Run a streaming claude -p task and return a multimodal RunResult.

    ``build_command(run_request) -> argv`` is injectable (tests pass a stub-
    pointing builder); by default the transport for the request's
    execution_location is used.
    """
    if build_command is None:
        build_command = build_command_for

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
    _reflect(status_path, RUN_STATE_RUNNING)

    argv = build_command(run_request)
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
    write_stream_json_message(build_user_message(run_request.input_content))

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

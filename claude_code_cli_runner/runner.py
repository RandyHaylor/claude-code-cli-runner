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
import threading
import time
import uuid

from .content import (
    build_initialize_control_request,
    build_injected_user_message,
    build_permission_control_response,
    build_user_message,
    extract_text_only,
    list_produced_artifacts,
    snapshot_workspace_files,
)
from .live_files import (
    CONTROL_END_AND_RETURN,
    CONTROL_PAUSE,
    CONTROL_PERMISSION_DECISION,
    CONTROL_RESUME,
    CONTROL_SEND_COMMAND,
    RUN_STATE_AWAITING_PERMISSION,
    RUN_STATE_IDLE_KILLED,
    RUN_STATE_OPERATOR_ENDED,
    RUN_STATE_TOKEN_LIMIT_HALTED,
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
from . import claude_session_store
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
    projects_root=None,
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

    # An EXPLICIT session id (resume-on-reply, collaborative turns) owns the
    # session lifecycle itself, so it never goes through the prime/fork reuse
    # path: run a single streaming turn whose argv carries --session-id/--resume.
    reuse = run_request.reusable_context
    if run_request.session_id:
        reuse = None
    if reuse is not None and run_request.enable_session_reuse:
        return _run_with_session_reuse(
            run_request,
            build_command=build_command,
            pause_poll_seconds=pause_poll_seconds,
            registry_path=registry_path,
            projects_root=projects_root,
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
    projects_root=None,
) -> RunResult:
    """Prime-once / fork-per-task path, with inline fallback on any failure.

    Cross-cwd reuse: the prime runs in the task's workspace and leaves a session
    jsonl under that cwd's claude-projects dir; its path is recorded in the
    registry. On a later fork whose task cwd differs from the prime cwd, the
    primed jsonl is COPIED into the task cwd's project dir before forking, so
    ``--resume`` finds it. Same-cwd is a no-op copy.
    """
    reuse = run_request.reusable_context
    chunk_id = reuse.chunk_id
    task_cwd = os.fspath(run_request.workspace_directory)

    def fall_back_inline(note: str) -> RunResult:
        return _stream_one_run(
            run_request,
            argv=build_command(run_request),
            input_content=_inline_input_content(run_request),
            pause_poll_seconds=pause_poll_seconds,
            startup_notes=[note],
        )

    try:
        record = session_registry.get_primed_record(
            chunk_id, registry_path=registry_path
        )
        primed_sid = record["session_id"] if record else None
        source_jsonl = record["source_jsonl"] if record else None
        notes = []
        if primed_sid is None:
            # PRIME ONCE: run a SIMPLE completing claude that ingests ONLY the
            # chunk (as a text prompt), record its id. Only text-only chunks can
            # be primed this way; a non-text chunk raises -> inline fallback.
            chunk_text = extract_text_only(list(reuse.content))
            if chunk_text is None:
                raise RuntimeError(
                    "reusable_context %r is not text-only; "
                    "simple priming unsupported (multimodal priming deferred)"
                    % chunk_id
                )
            # claude --session-id requires a valid UUID; the chunk_id<->session
            # association lives in the registry + startup notes, not in the id.
            primed_sid = str(uuid.uuid4())
            prime_argv = build_priming_claude_argv(run_request, primed_sid, chunk_text)
            _run_priming_session(run_request, argv=prime_argv)
            # The prime ran in the task workspace (prime_cwd == task_cwd here),
            # so claude wrote its jsonl under that cwd's project dir. Record both
            # the session id and the absolute jsonl path so later forks from a
            # DIFFERENT cwd can relocate it.
            source_jsonl = claude_session_store.session_jsonl_path(
                task_cwd, primed_sid, projects_root=projects_root
            )
            session_registry.record_primed_session_id(
                chunk_id,
                primed_sid,
                registry_path=registry_path,
                source_jsonl=source_jsonl,
            )
            notes.append(
                "reusable_context %r primed new session %s" % (chunk_id, primed_sid)
            )
        else:
            notes.append(
                "reusable_context %r reusing primed session %s" % (chunk_id, primed_sid)
            )

        # Cross-cwd reuse: claude --resume only finds the primed session if its
        # jsonl exists under the TASK cwd's project dir. Relocate it there first
        # (same-cwd is a no-op). A missing source jsonl raises -> inline fallback.
        if not source_jsonl:
            raise RuntimeError(
                "no source jsonl recorded for primed session %s" % primed_sid
            )
        claude_session_store.ensure_session_present_in_cwd(
            primed_sid, source_jsonl, task_cwd, projects_root=projects_root
        )

        # TASK as a FORK of the primed session: send ONLY the per-task remainder
        # (the chunk is already in the primed session, NOT re-sent here).
        task_sid = str(uuid.uuid4())  # must be a valid UUID for claude --session-id
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


def _run_priming_session(run_request: RunRequest, *, argv) -> None:
    """Run the SIMPLE, self-completing priming claude (``claude --session-id <sid>
    -p "<chunk text>"``) that leaves a reusable primed session behind, then
    exits. It takes NO stdin (the prompt is on argv), produces default output we
    do NOT parse, and is expected to exit 0 on its own.

    Not part of the live window. Raises on a non-zero exit, a timeout, or any
    launch failure so the caller can fall back to inline.
    """
    workspace_directory = os.fspath(run_request.workspace_directory)
    os.makedirs(workspace_directory, exist_ok=True)
    process = subprocess.Popen(
        argv,
        cwd=workspace_directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timeout = run_request.timeout_seconds or None
    try:
        _stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process(process)
        raise RuntimeError("priming claude timed out after %rs" % timeout)
    if process.returncode != 0:
        raise RuntimeError(
            "priming claude exited %r: %s"
            % (process.returncode, (stderr or "").strip())
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

    harness_environment = dict(os.environ)
    if run_request.run_environment_variables:
        harness_environment.update({
            str(name): str(value)
            for name, value in run_request.run_environment_variables.items()
        })
    process = subprocess.Popen(
        argv,
        cwd=workspace_directory,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=harness_environment,
    )

    def write_stream_json_message(message: dict) -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    # Live tool-permission escalation via the CLI's can_use_tool control protocol
    # is active only when a permission posture is set (transports launches the CLI
    # with --permission-prompt-tool stdio in that case). Send the one-time
    # initialize handshake the CLI expects BEFORE the prompt (as the Agent SDK does).
    permission_prompt_enabled = bool(getattr(run_request, "permission_mode", None))
    if permission_prompt_enabled:
        write_stream_json_message(build_initialize_control_request())

    # Deliver the multimodal prompt as a stdin stream-json user message. stdin
    # stays OPEN afterwards so mid-run send_command injection still works.
    write_stream_json_message(build_user_message(input_content))

    consumed_control_lines = 0
    paused = False
    operator_ended = False
    pending_permission_decision = None
    result_seen = False
    final_result_event = None
    collected_text_parts: list = []
    result_text_fallback: list = []
    deadline = (
        time.monotonic() + run_request.timeout_seconds
        if run_request.timeout_seconds
        else None
    )

    # IDLE-KILL WATCHDOG (the runner's explicit duty: kill a run whose harness
    # streams nothing for idle_kill_seconds — the caller only dictates the
    # policy value). Waiting on an operator permission decision and
    # operator-paused time are NOT idleness and suppress the watchdog.
    idle_watchdog_shared_state = {
        "last_stream_activity_monotonic": time.monotonic(),
        "idleness_suppressed": False,
        "run_finished": False,
        "idle_killed": False,
    }

    def watch_for_idle_harness_and_kill():
        idle_budget = float(run_request.idle_kill_seconds)
        while not idle_watchdog_shared_state["run_finished"]:
            time.sleep(min(1.0, idle_budget / 4))
            if idle_watchdog_shared_state["run_finished"]:
                return
            if idle_watchdog_shared_state["idleness_suppressed"]:
                # Permission wait / pause: reset the clock so post-wait time
                # is measured fresh.
                idle_watchdog_shared_state["last_stream_activity_monotonic"] = (
                    time.monotonic()
                )
                continue
            idle_for = time.monotonic() - idle_watchdog_shared_state[
                "last_stream_activity_monotonic"
            ]
            if idle_for > idle_budget:
                idle_watchdog_shared_state["idle_killed"] = True
                _reflect(status_path, RUN_STATE_IDLE_KILLED)
                _terminate_process(process)
                return

    if run_request.idle_kill_seconds:
        threading.Thread(
            target=watch_for_idle_harness_and_kill,
            name="runner-idle-kill-watchdog",
            daemon=True,
        ).start()

    # TOKEN ACCOUNTING (always collected + reported) and the HARD token-limit
    # halt (the runner's duty; the caller only supplies task_token_limit).
    # counted_tokens = input + output + cache_creation; cache READS excluded.
    cumulative_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "counted_tokens": 0,
    }
    usage_bearing_deltas_seen = 0
    token_limit_halted = False

    def _accumulate_usage_object(usage_object) -> None:
        nonlocal usage_bearing_deltas_seen
        if not isinstance(usage_object, dict):
            return
        for field_name in (
            "input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens",
        ):
            value = usage_object.get(field_name)
            if isinstance(value, (int, float)):
                cumulative_token_usage[field_name] += int(value)
        cumulative_token_usage["counted_tokens"] = (
            cumulative_token_usage["input_tokens"]
            + cumulative_token_usage["output_tokens"]
            + cumulative_token_usage["cache_creation_input_tokens"]
        )
        usage_bearing_deltas_seen += 1

    def accumulate_token_usage_from_chunk(chunk) -> None:
        """Per-turn usage arrives on the stream's message_delta events; the
        final result event's usage duplicates the last turn's, so it is used
        ONLY as a fallback when no delta ever carried usage."""
        if not isinstance(chunk, dict):
            return
        if chunk.get("type") == "stream_event":
            event = chunk.get("event") or {}
            if isinstance(event, dict) and event.get("type") == "message_delta":
                _accumulate_usage_object(event.get("usage"))
        elif chunk.get("type") == "result" and usage_bearing_deltas_seen == 0:
            _accumulate_usage_object(chunk.get("usage"))

    def token_limit_reached() -> bool:
        if not run_request.task_token_limit:
            return False
        return (
            cumulative_token_usage["counted_tokens"]
            >= int(run_request.task_token_limit)
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
            accumulate_token_usage_from_chunk(chunk)
            if isinstance(chunk, dict) and chunk.get("type") == "result":
                result_seen = True
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        os.fsync(log_handle.fileno())

    def drain_control_channel() -> bool:
        nonlocal consumed_control_lines, paused, operator_ended
        nonlocal pending_permission_decision
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
            elif kind == CONTROL_PERMISSION_DECISION:
                # The operator's allow/deny for a pending tool-permission request;
                # the awaiting loop below picks it up and writes the control_response.
                pending_permission_decision = intent.get("decision")
            elif kind == CONTROL_END_AND_RETURN:
                operator_ended = True
                _reflect(status_path, RUN_STATE_OPERATOR_ENDED)
                return True
        return False

    def await_permission_decision_and_respond(control_request_chunk) -> bool:
        """Hold the run while the agent waits for the operator's permission
        decision on a ``can_use_tool`` request, then write the ``control_response``
        to stdin so the tool runs (allow) or is blocked (deny). Returns True if the
        operator ENDED the run during the wait (caller should stop)."""
        nonlocal pending_permission_decision
        request = control_request_chunk.get("request", {}) or {}
        request_id = control_request_chunk.get("request_id")
        # Surface the request as a DISTINCT live-log record the dashboard renders
        # with approve/deny controls.
        log_handle.write(
            json.dumps(
                {
                    "received_at": time.time(),
                    "permission_request": {
                        "request_id": request_id,
                        "tool_name": request.get("tool_name"),
                        "input": request.get("input"),
                        "tool_use_id": request.get("tool_use_id"),
                        "decision_reason": request.get("decision_reason"),
                        "permission_suggestions": request.get("permission_suggestions"),
                    },
                }
            )
            + "\n"
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
        _reflect(status_path, RUN_STATE_AWAITING_PERMISSION)
        # Waiting on a HUMAN decision is not harness idleness — suppress the
        # idle-kill watchdog for the duration of the wait.
        idle_watchdog_shared_state["idleness_suppressed"] = True
        pending_permission_decision = None
        try:
            while True:
                if drain_control_channel():
                    return True  # operator ended the run while we awaited the decision
                if pending_permission_decision is not None:
                    decision = pending_permission_decision or {}
                    pending_permission_decision = None
                    behavior = decision.get("behavior", "deny")
                    write_stream_json_message(
                        build_permission_control_response(
                            request_id,
                            behavior,
                            updated_input=decision.get("updated_input")
                            or request.get("input"),
                            message=decision.get("message"),
                        )
                    )
                    _record_permission_resolved(request_id, behavior)
                    _reflect(status_path, RUN_STATE_RUNNING)
                    return False
                if deadline is not None and time.monotonic() > deadline:
                    # Timed out — deny so the run can finish rather than hang forever.
                    write_stream_json_message(
                        build_permission_control_response(
                            request_id, "deny",
                            message="Timed out awaiting the operator's decision.",
                        )
                    )
                    _record_permission_resolved(request_id, "deny")
                    _reflect(status_path, RUN_STATE_RUNNING)
                    return False
                time.sleep(pause_poll_seconds)
        finally:
            idle_watchdog_shared_state["idleness_suppressed"] = False

    def _record_permission_resolved(request_id, behavior) -> None:
        """Mark a permission request resolved IN THE LIVE LOG (which is teed to
        the host), so a reader can tell a pending request from a decided one
        without depending on the resource-local run-state sidecar."""
        log_handle.write(
            json.dumps(
                {
                    "received_at": time.time(),
                    "permission_resolved": {
                        "request_id": request_id,
                        "behavior": behavior,
                    },
                }
            )
            + "\n"
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())

    captured_harness_stderr = ""
    try:
        for raw_line in process.stdout:
            idle_watchdog_shared_state["last_stream_activity_monotonic"] = (
                time.monotonic()
            )
            raw_line = raw_line.rstrip("\n")
            if raw_line == "":
                continue

            if drain_control_channel():
                break

            if paused:
                idle_watchdog_shared_state["idleness_suppressed"] = True
                while paused:
                    if drain_control_channel():
                        break
                    time.sleep(pause_poll_seconds)
                idle_watchdog_shared_state["idleness_suppressed"] = False
            if operator_ended:
                break

            if deadline is not None and time.monotonic() > deadline:
                break

            # Control-protocol lines exist only with --permission-prompt-tool stdio
            # (permission_prompt_enabled). A can_use_tool request HOLDS the run
            # awaiting the operator's allow/deny; other control_response/keep_alive/
            # cancel lines are protocol bookkeeping — recorded, not rendered.
            if permission_prompt_enabled:
                try:
                    protocol_chunk = json.loads(raw_line)
                except (json.JSONDecodeError, ValueError):
                    protocol_chunk = None
                if isinstance(protocol_chunk, dict):
                    protocol_type = protocol_chunk.get("type")
                    if protocol_type == "control_request" and (
                        protocol_chunk.get("request") or {}
                    ).get("subtype") == "can_use_tool":
                        if await_permission_decision_and_respond(protocol_chunk):
                            break
                        continue
                    if protocol_type in (
                        "control_response",
                        "control_cancel_request",
                        "keep_alive",
                    ):
                        log_handle.write(
                            json.dumps(
                                {"received_at": time.time(), "protocol": protocol_chunk}
                            )
                            + "\n"
                        )
                        log_handle.flush()
                        os.fsync(log_handle.fileno())
                        continue

            append_chunk_to_live_log(raw_line)

            if token_limit_reached() and not result_seen:
                # HARD HALT (the runner's duty): the run consumed its whole
                # token budget — stop the harness NOW and report it as a coded
                # halt for the caller's escalation flow.
                token_limit_halted = True
                _reflect(status_path, RUN_STATE_TOKEN_LIMIT_HALTED)
                log_handle.write(
                    json.dumps({
                        "received_at": time.time(),
                        "runner_token_limit_halt": {
                            "task_token_limit": run_request.task_token_limit,
                            "token_usage": dict(cumulative_token_usage),
                            "note": "halted due to token limit for task reached",
                        },
                    }) + "\n"
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
                _terminate_process(process)
                break

            if result_seen:
                break
    finally:
        idle_watchdog_shared_state["run_finished"] = True
        if idle_watchdog_shared_state["idle_killed"]:
            # Make the kill visible in the teed live log, not just the sidecar.
            log_handle.write(
                json.dumps({
                    "received_at": time.time(),
                    "runner_idle_kill": {
                        "idle_kill_seconds": run_request.idle_kill_seconds,
                        "note": "runner killed the harness process: no stream "
                                "activity within the idle budget",
                    },
                }) + "\n"
            )
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
        # Drain the harness's stderr so a failed run can report WHY it failed (an
        # error_during_execution result carries no message). Best-effort: the pipe
        # was always opened but never read, so the real error text was lost.
        try:
            if process.stderr is not None:
                captured_harness_stderr = (process.stderr.read() or "").strip()
        except (OSError, ValueError):
            captured_harness_stderr = ""

    final_run_state = RUN_STATE_OPERATOR_ENDED if operator_ended else RUN_STATE_RUNNING
    if not operator_ended:
        # Reflect a terminal "running->done" by leaving running; the sidecar's
        # job is the live annotation, and the result carries the real outcome.
        final_run_state = read_run_state_value(status_path) or RUN_STATE_RUNNING
    if idle_watchdog_shared_state["idle_killed"]:
        final_run_state = RUN_STATE_IDLE_KILLED
        if not captured_harness_stderr:
            captured_harness_stderr = (
                "runner killed the harness process: no stream activity within "
                "the idle budget (%.0fs)" % float(run_request.idle_kill_seconds)
            )
    if token_limit_halted:
        final_run_state = RUN_STATE_TOKEN_LIMIT_HALTED
        if not captured_harness_stderr:
            captured_harness_stderr = (
                "halted due to token limit for task reached: counted %d of "
                "limit %d tokens"
                % (
                    cumulative_token_usage["counted_tokens"],
                    int(run_request.task_token_limit),
                )
            )

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
        harness_stderr=captured_harness_stderr,
        token_usage=dict(cumulative_token_usage),
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

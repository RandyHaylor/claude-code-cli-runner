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

import dataclasses
import json
import os
import signal
import subprocess
import threading
import time
import uuid

from .content import (
    build_initialize_control_request,
    build_permission_control_response,
    extract_text_only,
    list_produced_artifacts,
    snapshot_workspace_files,
)
from .keep_alive_registry import (
    clear_keep_alive_record,
    record_keep_alive_signal,
    seconds_since_last_keep_alive,
)
from .live_files import (
    CONTROL_END_AND_RETURN,
    CONTROL_PAUSE,
    CONTROL_PERMISSION_DECISION,
    CONTROL_RESUME,
    CONTROL_SEND_COMMAND,
    RUN_STATE_AWAITING_PERMISSION,
    RUN_STATE_IDLE_KILLED,
    RUN_STATE_KEEP_ALIVE_LOST_KILLED,
    RUN_STATE_OPERATOR_ENDED,
    RUN_STATE_TOKEN_LIMIT_HALTED,
    RUN_STATE_PAUSED,
    RUN_STATE_RUNNING,
    control_channel_path,
    live_log_path,
    read_new_control_intents,
    run_status_path,
)
from .harness_activity_events import derive_harness_activity_event
from .harness_integration import get_harness_integration
from .request import RunRequest, TextBlock
from .runner_provided_mcp_registry import (
    DigestibleToolNameUnknown,
    split_assigned_tools_into_builtins_and_runner_mcps,
    translate_digestible_tool_name_to_mcp,
)
from .unharness_tool_call_detection import (
    compose_tool_result_turn_message_text,
    detect_unharness_tool_calls_in_turn_text,
)
from .unharness_mcp_tool_executor import execute_unharness_mcp_tool_call
from .result import RunResult
from . import session_registry
from . import claude_session_store
from . import claude_base_session
from .transports import build_command_for
from .claude_harness import (
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
    # A test injects a stub ``build_command``; base-session forking (which builds its own
    # prime/fork argv, bypassing the stub) must NOT fire in that case, or a stub test
    # would launch the real prime/fork flow. So forking is a PRODUCTION-path behavior:
    # only when build_command is the default (not injected).
    build_command_was_injected = build_command is not None
    if build_command is None:
        build_command = build_command_for

    # The prime-once/fork-per-task session-reuse, resume-failure fallback, and
    # warm-base-fork paths ALL depend on a harness that supports caller-chosen
    # session ids + an on-disk session store (declared as one capability). A
    # harness without it simply prepends the chunk inline (always correct, just
    # no cache reuse) and runs a single plain turn.
    harness_supports_session_prime_and_fork = get_harness_integration(
        run_request.harness
    ).capabilities.supports_session_prime_and_fork

    # An EXPLICIT session id (resume-on-reply, collaborative turns) owns the
    # session lifecycle itself, so it never goes through the prime/fork reuse
    # path: run a single streaming turn whose argv carries --session-id/--resume.
    reuse = run_request.reusable_context
    if run_request.session_id:
        reuse = None
    if reuse is not None and not harness_supports_session_prime_and_fork:
        reuse = None
    if reuse is not None and run_request.enable_session_reuse:
        return _run_with_session_reuse(
            run_request,
            build_command=build_command,
            pause_poll_seconds=pause_poll_seconds,
            registry_path=registry_path,
            projects_root=projects_root,
        )

    # RESUME-FAILURE FALLBACK (raw-1216/1222): a claude RESUME turn whose session
    # transcript is GONE from the task cwd (e.g. VM /tmp cleared) cannot be resumed.
    # Deterministic pre-check (no stderr parsing): if the transcript jsonl is missing,
    # run a FRESH session with the caller-supplied full fallback prompt instead of
    # failing; claude mints the new id and it is captured on the normal result path.
    # Gated on the session-prime/fork capability (adapter isolation); requires the
    # caller to have sent the prompt.
    if (
        harness_supports_session_prime_and_fork
        and run_request.resume_session
        and run_request.session_id
        and run_request.resume_fallback_prompt
    ):
        resumed_transcript_path = claude_session_store.session_jsonl_path(
            os.fspath(run_request.workspace_directory),
            run_request.session_id,
            projects_root=projects_root,
        )
        if not os.path.isfile(resumed_transcript_path):
            fresh_session_request = dataclasses.replace(
                run_request,
                session_id=None,
                resume_session=False,
                input_content=[TextBlock(run_request.resume_fallback_prompt)],
            )
            return _stream_one_run(
                fresh_session_request,
                argv=build_command(fresh_session_request),
                input_content=list(fresh_session_request.input_content),
                pause_poll_seconds=pause_poll_seconds,
                startup_notes=[
                    "resume of session %s impossible (transcript missing at %s); "
                    "started a FRESH session with the resume fallback prompt"
                    % (run_request.session_id, resumed_transcript_path)
                ],
            )

    # BASE-SESSION FORK (raw-1211/1212): a claude task's FIRST run (a caller-minted
    # session_id, resume_session False, no reusable chunk) forks its fresh session from
    # a warm, dated base session whose universal init prompt is already cached — instead
    # of paying the full session-creation cost. CLAUDE-ONLY (adapter isolation),
    # env-gated (default on), best-effort: any failure falls through to a plain run.
    if (
        harness_supports_session_prime_and_fork
        and not run_request.resume_session
        and not build_command_was_injected
        and _claude_base_session_fork_enabled()
    ):
        forked_result = _try_fork_task_from_base_session(
            run_request,
            pause_poll_seconds=pause_poll_seconds,
            projects_root=projects_root,
        )
        if forked_result is not None:
            return forked_result

    # No reuse / no fork: inline the chunk (if present) and run normally.
    effective_input = _inline_input_content(run_request)
    return _stream_one_run(
        run_request,
        argv=build_command(run_request),
        input_content=effective_input,
        pause_poll_seconds=pause_poll_seconds,
        startup_notes=None,
    )


def _claude_base_session_fork_enabled() -> bool:
    """Base-session forking is ON by default (raw-1211/1212); an operator can disable it
    with ``UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK=0`` (0/false/no, case-insensitive)."""
    raw = os.environ.get("UNHARNESS_ENABLE_CLAUDE_BASE_SESSION_FORK")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _try_fork_task_from_base_session(
    run_request: RunRequest,
    *,
    pause_poll_seconds: float,
    projects_root=None,
) -> "RunResult | None":
    """Fork this task's fresh session (``run_request.session_id``) from the warm base
    session. Returns the RunResult on success, or None on ANY failure so the caller
    runs the task as a plain new session (work is never blocked)."""
    task_cwd = os.fspath(run_request.workspace_directory)
    try:
        def prime_base_session(base_session_id: str) -> None:
            _run_priming_session(
                run_request,
                argv=build_priming_claude_argv(
                    run_request,
                    base_session_id,
                    claude_base_session.UNIVERSAL_INIT_PROMPT,
                ),
            )

        def session_jsonl_path_for(base_session_id: str) -> str:
            return claude_session_store.session_jsonl_path(
                task_cwd, base_session_id, projects_root=projects_root
            )

        base_record = claude_base_session.ensure_fresh_base_session(
            prime_base_session=prime_base_session,
            session_jsonl_path_for=session_jsonl_path_for,
            unharness_idle_seconds=run_request.unharness_idle_seconds,
        )
        base_session_id = base_record["session_id"]
        base_source_jsonl = base_record["source_jsonl"]
        if not base_source_jsonl:
            raise RuntimeError("base session has no recorded source jsonl")
        # claude --resume only finds the base if its jsonl exists under THIS task's cwd.
        claude_session_store.ensure_session_present_in_cwd(
            base_session_id, base_source_jsonl, task_cwd, projects_root=projects_root
        )
        fork_argv = build_fork_claude_argv(run_request, base_session_id)
        return _stream_one_run(
            run_request,
            argv=fork_argv,
            input_content=list(run_request.input_content),
            pause_poll_seconds=pause_poll_seconds,
            startup_notes=[
                "forked a new task session from base session %s "
                "(claude mints the forked id; captured from the result)"
                % base_session_id
            ],
        )
    except Exception:  # noqa: BLE001 — base forking must NEVER fail the task
        # A base that could not be forked is likely unusable; drop it so the next
        # task primes a fresh one.
        try:
            claude_base_session.forget_base_session()
        except Exception:  # noqa: BLE001
            pass
        return None


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
        # (the chunk is already in the primed session, NOT re-sent here). claude mints
        # the forked session id; we do not choose it.
        fork_argv = build_fork_claude_argv(run_request, primed_sid)
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

    # A fresh run starts a clean live log (so a tailer's offsets are
    # meaningful); a run that RESUMES an existing session APPENDS — the resumed
    # session's history is environment, not something to rebuild.
    if not run_request.resume_session:
        open(log_path, "w", encoding="utf-8").close()
    # The control channel is TRANSPORT for exactly ONE run: a command written
    # for a previous run (e.g. a stale end_and_return from a manual stop) must
    # never be consumed by this run, so the channel starts empty.
    open(control_path, "w", encoding="utf-8").close()
    if startup_notes:
        with open(log_path, "a", encoding="utf-8") as note_handle:
            for note in startup_notes:
                note_handle.write(
                    json.dumps({
                        "received_at": time.time(),
                        "activity": {"kind": "runner_note", "text": note},
                        "runner_note": note,
                    }) + "\n"
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
    def launch_harness_subprocess(subprocess_argv):
        # OWN PROCESS GROUP (raw-821): the harness and every subprocess it
        # spawns live in one killable group, so terminating a run can never
        # leave stray harness children churning after the runner is gone.
        return subprocess.Popen(
            subprocess_argv,
            cwd=workspace_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=harness_environment,
            start_new_session=True,
        )

    process = launch_harness_subprocess(argv)
    # The CURRENT turn's harness process. A per-turn harness (e.g. pi) SWAPS this
    # when it continues the session for a follow-up turn; the watchdogs, control
    # channel, kill, and stderr capture all read the live process via this holder
    # so they follow the swap. A single-streaming harness (claude) never swaps it.
    active_harness_process_holder = {"process": process}

    def write_stream_json_message(message: dict) -> None:
        live_process = active_harness_process_holder["process"]
        if live_process.stdin is None:
            return
        try:
            live_process.stdin.write(json.dumps(message) + "\n")
            live_process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    # The harness integration owns per-run output normalization (identity for a
    # harness whose stdout already IS the internal chunk shape) and how the prompt
    # is delivered on that harness's own terms.
    harness_integration = get_harness_integration(run_request.harness)
    output_event_normalizer = harness_integration.create_output_event_normalizer()

    # Live tool-permission escalation via the CLI's can_use_tool control protocol
    # is active only when a permission posture is set (transports launches the CLI
    # with --permission-prompt-tool stdio in that case). Send the one-time
    # initialize handshake the CLI expects BEFORE the prompt (as the Agent SDK does).
    permission_prompt_enabled = bool(getattr(run_request, "permission_mode", None))
    if permission_prompt_enabled:
        write_stream_json_message(build_initialize_control_request())

    harness_integration.deliver_prompt(
        process=process,
        input_content=input_content,
        run_request=run_request,
        write_stream_json_message=write_stream_json_message,
        permission_prompt_enabled=permission_prompt_enabled,
    )

    consumed_control_lines = 0
    paused = False
    operator_ended = False
    pending_permission_decision = None
    result_seen = False
    final_result_event = None
    # The session id the harness reports (uniform contract: any harness's
    # normalized chunk carries it under "session_id" — claude's system-init /
    # result events, pi's session event). Captured from the FIRST chunk that
    # carries one, so it survives runs that never reach a final result event
    # (aborted / operator-ended runs).
    harness_reported_session_id = ""
    collected_text_parts: list = []
    result_text_fallback: list = []
    deadline = (
        time.monotonic() + run_request.timeout_seconds
        if run_request.timeout_seconds
        else None
    )

    # RUNNER-MEDIATED MCP TOOL LOOP (nd-472/nd-474/nd-476): enabled when the
    # session's assigned tool list names a runner-provided MCP (e.g. "unharness-api").
    # At each turn's stop (the `result` event — the existing stop, nd-476), the turn's
    # text is checked for a detected tool call: if found the stop is WITHHELD, the
    # call is translated (digestible -> standard MCP, nd-473) and executed, and the
    # result is fed back over stdin as the next turn's input — the agent keeps going.
    # A turn with no tool call ends the run exactly as before.
    _, session_runner_provided_mcp_names = (
        split_assigned_tools_into_builtins_and_runner_mcps(
            run_request.assigned_tool_list or []
        )
    )
    tool_loop_enabled = bool(session_runner_provided_mcp_names)
    active_runner_mcp_name = (
        session_runner_provided_mcp_names[0] if session_runner_provided_mcp_names else None
    )
    execute_tool_call = (
        run_request.unharness_tool_call_executor or execute_unharness_mcp_tool_call
    )
    current_turn_text_start_index = 0
    completed_tool_round_trips = 0
    # Runaway guard — never silent: hitting the cap is logged as a runner note.
    TOOL_ROUND_TRIP_MAXIMUM = 25

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
                _terminate_process(active_harness_process_holder["process"])
                return

    if run_request.idle_kill_seconds:
        threading.Thread(
            target=watch_for_idle_harness_and_kill,
            name="runner-idle-kill-watchdog",
            daemon=True,
        ).start()

    # KEEP-ALIVE FAIL-SAFE (raw-830): the orchestrator relays its heartbeat to
    # this runner (~every 10s) for the task WHILE IT IS IN PROGRESS. When the
    # relayed heartbeat stops for keep_alive_timeout_seconds — orchestrator
    # dead, task wiped, or task no longer in progress — the run is AGGRESSIVELY
    # killed (straight SIGKILL of the process group), so an orphaned harness
    # can never keep consuming resources.
    keep_alive_shared_state = {"keep_alive_killed": False}
    keep_alive_task_id = run_request.keep_alive_task_id or os.path.basename(
        workspace_directory.rstrip("/")
    )

    def watch_for_lost_keep_alive_and_kill():
        timeout_seconds = float(run_request.keep_alive_timeout_seconds)
        while not idle_watchdog_shared_state["run_finished"]:
            time.sleep(2.0)
            if idle_watchdog_shared_state["run_finished"]:
                return
            signal_age = seconds_since_last_keep_alive(keep_alive_task_id)
            if signal_age is not None and signal_age > timeout_seconds:
                keep_alive_shared_state["keep_alive_killed"] = True
                _reflect(status_path, RUN_STATE_KEEP_ALIVE_LOST_KILLED)
                live_process = active_harness_process_holder["process"]
                try:
                    os.killpg(live_process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    live_process.kill()
                return

    if run_request.keep_alive_expected:
        # The dispatch itself counts as the first signal.
        record_keep_alive_signal(keep_alive_task_id)
        threading.Thread(
            target=watch_for_lost_keep_alive_and_kill,
            name="runner-keep-alive-watchdog",
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

    def record_chunk_and_update_run_bookkeeping(chunk) -> None:
        nonlocal result_seen, harness_reported_session_id
        record = {"received_at": time.time(), "chunk": chunk}
        # STANDARD activity facet (raw-780/781): renderers read ONLY this,
        # harness-agnostic; the raw chunk stays alongside for troubleshooting.
        activity = derive_harness_activity_event(chunk)
        if activity is not None:
            record["activity"] = activity
        render_text_from_chunk(chunk)
        accumulate_token_usage_from_chunk(chunk)
        if not harness_reported_session_id and isinstance(chunk, dict):
            chunk_session_id = chunk.get("session_id")
            if isinstance(chunk_session_id, str) and chunk_session_id:
                harness_reported_session_id = chunk_session_id
        if isinstance(chunk, dict) and chunk.get("type") == "result":
            result_seen = True
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        os.fsync(log_handle.fileno())

    def append_chunk_to_live_log(raw_line: str):
        try:
            chunk = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            chunk = None
        if chunk is None:
            record = {"received_at": time.time(), "raw": raw_line}
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            os.fsync(log_handle.fileno())
            return
        for normalized_chunk in output_event_normalizer.normalize(chunk):
            record_chunk_and_update_run_bookkeeping(normalized_chunk)

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
                # Hand the GENERIC operator command DOWN to the harness integration,
                # which translates it to that harness's wire form (claude/opencode: a
                # stream-json injected user message; pi: an RPC steer). The core stays
                # harness-agnostic — no harness-specific message shape here (paradigm:
                # zero `if harness == ...`). No-op for a harness lacking the hook.
                inject_hook = getattr(
                    harness_integration, "deliver_injected_command", None
                )
                if callable(inject_hook):
                    try:
                        inject_hook(
                            process=active_harness_process_holder["process"],
                            command_text=intent.get("command_text", ""),
                            write_stream_json_message=write_stream_json_message,
                        )
                    except Exception:
                        pass
            elif kind == CONTROL_PERMISSION_DECISION:
                # The operator's allow/deny for a pending tool-permission request;
                # the awaiting loop below picks it up and writes the control_response.
                pending_permission_decision = intent.get("decision")
            elif kind == CONTROL_END_AND_RETURN:
                # GRACEFUL ABORT FIRST (optional per-harness hook, e.g. pi RPC's
                # {"type":"abort"}): cancel the in-flight upstream generation cleanly
                # BEFORE the process is terminated — this frees the backend immediately
                # (verified: llama.cpp stops ~0.03s after a pi abort) without relying on
                # the OS-kill of _terminate_process. No-op for harnesses without the hook.
                abort_hook = getattr(harness_integration, "request_abort", None)
                if callable(abort_hook):
                    try:
                        abort_hook(
                            process=active_harness_process_holder["process"],
                            write_stream_json_message=write_stream_json_message,
                        )
                    except Exception:
                        pass
                    # BOUNDED POST-ABORT READ (generic contract): a normalizer
                    # that declares ``final_usage_after_abort_reported = False``
                    # has final output still coming after the abort (e.g. the
                    # aborted run's token stats — the only usage report a
                    # mid-turn abort ever gets). Keep reading its output until
                    # it flips the flag, EOF, or the deadline; each line goes
                    # through the normal chunk pipeline so usage accumulates.
                    if (
                        getattr(
                            output_event_normalizer,
                            "final_usage_after_abort_reported",
                            None,
                        )
                        is False
                    ):
                        post_abort_read_deadline = time.monotonic() + 5.0
                        while (
                            output_event_normalizer.final_usage_after_abort_reported
                            is False
                            and time.monotonic() < post_abort_read_deadline
                        ):
                            try:
                                post_abort_raw_line = next(
                                    current_turn_line_iterator
                                )
                            except (StopIteration, ValueError, OSError):
                                break
                            post_abort_raw_line = post_abort_raw_line.rstrip("\n")
                            if post_abort_raw_line:
                                append_chunk_to_live_log(post_abort_raw_line)
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
                    "activity": {
                        "kind": "permission_request",
                        "request_id": request_id,
                        "tool_name": request.get("tool_name"),
                    },
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
                    "activity": {
                        "kind": "permission_resolved",
                        "request_id": request_id,
                        "behavior": behavior,
                    },
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
    # Read the run's output as a sequence of TURNS. Each line comes from the
    # CURRENT turn's process; when the runner-mediated tool loop serves a tool and
    # continues the session, the harness's deliver_followup_turn hands back the
    # process to keep reading (the same streaming process for claude, a fresh
    # per-turn process for pi), and we swap the line iterator to it. The loop body
    # is harness-agnostic; only deliver_followup_turn knows the mechanism.
    current_turn_line_iterator = iter(active_harness_process_holder["process"].stdout)
    try:
        while True:
            try:
                raw_line = next(current_turn_line_iterator)
            except StopIteration:
                break
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
                        "activity": {
                            "kind": "runner_note",
                            "text": "halted due to token limit for task reached",
                        },
                        "runner_token_limit_halt": {
                            "task_token_limit": run_request.task_token_limit,
                            "token_usage": dict(cumulative_token_usage),
                            "note": "halted due to token limit for task reached",
                        },
                    }) + "\n"
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
                _terminate_process(active_harness_process_holder["process"])
                break

            if result_seen:
                if not tool_loop_enabled:
                    break
                just_completed_turn_text = "".join(
                    collected_text_parts[current_turn_text_start_index:]
                )
                detected_tool_calls = detect_unharness_tool_calls_in_turn_text(
                    just_completed_turn_text
                )
                if not detected_tool_calls:
                    break  # a genuine final message: the stop propagates as before
                if completed_tool_round_trips >= TOOL_ROUND_TRIP_MAXIMUM:
                    log_handle.write(json.dumps({
                        "received_at": time.time(),
                        "activity": {
                            "kind": "runner_note",
                            "text": "unharness tool round-trip maximum (%d) reached; "
                                    "ending the run" % TOOL_ROUND_TRIP_MAXIMUM,
                        },
                    }) + "\n")
                    log_handle.flush()
                    os.fsync(log_handle.fileno())
                    break
                # WITHHOLD the stop: execute each detected call, feed the results
                # back as the next turn's input, and keep the agent going.
                tool_result_message_texts = []
                for detected_call in detected_tool_calls:
                    digestible_tool_name = detected_call["tool"]
                    try:
                        mcp_tool_name = translate_digestible_tool_name_to_mcp(
                            active_runner_mcp_name, digestible_tool_name
                        )
                        result_text, tool_call_errored = execute_tool_call(
                            mcp_tool_name,
                            detected_call["arguments"],
                            run_request.run_environment_variables,
                        )
                    except DigestibleToolNameUnknown as unknown_tool:
                        result_text, tool_call_errored = str(unknown_tool), True
                    tool_result_message_texts.append(
                        compose_tool_result_turn_message_text(
                            digestible_tool_name, result_text, tool_call_errored
                        )
                    )
                    log_handle.write(json.dumps({
                        "received_at": time.time(),
                        "activity": {
                            "kind": "runner_note",
                            "text": "executed unharness tool call %r (error=%s); "
                                    "stop withheld, result fed back"
                                    % (digestible_tool_name, tool_call_errored),
                        },
                    }) + "\n")
                    log_handle.flush()
                    os.fsync(log_handle.fileno())
                followup_message_text = "\n\n".join(tool_result_message_texts)
                followup_session_id = (
                    final_result_event.get("session_id")
                    if isinstance(final_result_event, dict)
                    else None
                )
                # Continue the session via the harness's OWN mechanism (shared
                # surface): claude injects on its open stdin and returns the same
                # process; pi runs a fresh `pi --session <id>` turn and returns the
                # new process. The loop just keeps reading whatever it hands back.
                followup_process = harness_integration.deliver_followup_turn(
                    current_process=active_harness_process_holder["process"],
                    message_text=followup_message_text,
                    session_id=followup_session_id,
                    run_request=run_request,
                    launch_harness_subprocess=launch_harness_subprocess,
                )
                active_harness_process_holder["process"] = followup_process
                current_turn_line_iterator = iter(followup_process.stdout)
                completed_tool_round_trips += 1
                result_seen = False
                final_result_event = None
                current_turn_text_start_index = len(collected_text_parts)
                continue
    finally:
        idle_watchdog_shared_state["run_finished"] = True
        if run_request.keep_alive_expected:
            clear_keep_alive_record(keep_alive_task_id)
        if keep_alive_shared_state["keep_alive_killed"]:
            log_handle.write(
                json.dumps({
                    "received_at": time.time(),
                    "activity": {
                        "kind": "runner_note",
                        "text": "runner killed the harness process: the "
                                "orchestrator's keep-alive heartbeat stopped",
                    },
                    "runner_keep_alive_lost_kill": {
                        "keep_alive_timeout_seconds":
                            run_request.keep_alive_timeout_seconds,
                        "note": "no relayed keep-alive heartbeat within the "
                                "timeout; run treated as orphaned and killed",
                    },
                }) + "\n"
            )
        if idle_watchdog_shared_state["idle_killed"]:
            # Make the kill visible in the teed live log, not just the sidecar.
            log_handle.write(
                json.dumps({
                    "received_at": time.time(),
                    "activity": {
                        "kind": "runner_note",
                        "text": "runner killed the harness process: idle budget exceeded",
                    },
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
        final_turn_process = active_harness_process_holder["process"]
        if operator_ended or result_seen:
            _terminate_process(final_turn_process)
        else:
            try:
                exit_code = final_turn_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _terminate_process(final_turn_process)
        if exit_code is None:
            exit_code = final_turn_process.poll()
        # Drain the harness's stderr so a failed run can report WHY it failed (an
        # error_during_execution result carries no message). Best-effort: the pipe
        # was always opened but never read, so the real error text was lost.
        try:
            if final_turn_process.stderr is not None:
                captured_harness_stderr = (final_turn_process.stderr.read() or "").strip()
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
    if keep_alive_shared_state["keep_alive_killed"]:
        final_run_state = RUN_STATE_KEEP_ALIVE_LOST_KILLED
        if not captured_harness_stderr:
            captured_harness_stderr = (
                "runner killed the harness process: no keep-alive heartbeat "
                "relayed within %.0fs — run treated as orphaned"
                % float(run_request.keep_alive_timeout_seconds)
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
        harness_session_id=harness_reported_session_id,
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
    """Best-effort terminate-then-kill a still-running harness and reap it.

    Signals the WHOLE PROCESS GROUP (the harness is spawned with
    ``start_new_session=True``), so tool subprocesses and any children the
    harness forked die with it — a terminated run must never leave stray
    harness processes running (raw-821)."""
    if process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()
            process.wait(timeout=5)
    except (OSError, ValueError):
        pass

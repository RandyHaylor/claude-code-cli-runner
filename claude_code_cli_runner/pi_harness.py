"""The Pi CLI harness integration — a fully isolated implementation of
``HarnessIntegration`` on Pi's own terms, NOT modeled on any other harness.

Pi (``@earendil-works/pi-coding-agent``) is driven non-interactively in RPC mode:
``pi --mode rpc`` keeps the process ALIVE with stdin OPEN, reading JSONL commands
(one per LF) and streaming the SAME LF-delimited JSON events json mode does, AS THEY
HAPPEN. Verified behavior (pi 0.74.2):
  * The prompt is a JSONL command on STDIN: ``{"id":..,"type":"prompt","message":..}``;
    stdin STAYS OPEN so follow-up turns and a graceful ``{"type":"abort"}`` can be sent
    mid-run. Command acknowledgements arrive as ``{"type":"response",..}`` objects
    (dropped by the normalizer — they are not agent events).
  * A fresh run (no --session) makes Pi MINT its own session id, reported on the
    first ``{"type":"session","id":...}`` event; a later turn resumes it with
    ``--session <id>``. There is no caller-chosen create id.
  * Final assistant text + token usage per turn arrive on ``turn_end``; the run's
    terminal marker is ``agent_end``.
  * ``{"type":"abort"}`` cancels the current operation WITHOUT killing the process —
    verified live (2026-07-20) to stop the upstream llama.cpp generation ~0.03s later.

Pi is a reduced-capability harness: no operator permission protocol, mints its
own session ids, no prime/fork reuse, text input only; RPC mode keeps stdin open.

Deployment configuration is taken from environment (the documented way a run is
handed its tooling facts), never hard-coded:
  PI_COMMAND                    the pi executable (default "pi")
  PI_PROVIDER                   fallback provider when the model id has no
                                "<provider>/<id>" prefix (default "ollama")
  PI_SESSION_DIR                session storage dir (default <workspace>/pi_sessions)
  PI_EXTENSION_PATHS            os.pathsep-separated extension files to load with -e
The model string may be "<provider>/<model-id>" (e.g. "ollama/gemma4-…"): the
prefix becomes --provider and the remainder --model.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import List

from .harness_integration import (
    HarnessCapabilities,
    PromptDeliveryOutcome,
    register_harness_integration,
)
from .request import TextBlock
from .runner_provided_mcp_registry import (
    split_assigned_tools_into_builtins_and_runner_mcps,
)

HARNESS_ID_PI = "pi"

PI_COMMAND_ENVIRONMENT_VARIABLE = "PI_COMMAND"
PI_PROVIDER_ENVIRONMENT_VARIABLE = "PI_PROVIDER"
PI_SESSION_DIRECTORY_ENVIRONMENT_VARIABLE = "PI_SESSION_DIR"
PI_EXTENSION_PATHS_ENVIRONMENT_VARIABLE = "PI_EXTENSION_PATHS"

DEFAULT_PI_PROVIDER = "ollama"


def _fresh_rpc_command_id() -> str:
    """A short unique id echoed back on the command's response for correlation."""
    return uuid.uuid4().hex


def write_pi_rpc_command_line(process, command: dict) -> None:
    """Write ONE RPC command as a single LF-terminated JSON line to pi's stdin (strict
    JSONL framing — exactly one command per ``\\n``). Best-effort: a closed/broken stdin
    is swallowed so a control action (abort / follow-up) never crashes the run."""
    if process is None or getattr(process, "stdin", None) is None:
        return
    try:
        process.stdin.write(json.dumps(command) + "\n")
        process.stdin.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass


def split_model_string_into_provider_and_model(model_string, fallback_provider):
    """"<provider>/<model-id>" -> (provider, model-id); a bare id keeps the
    fallback provider. Only the FIRST "/" splits, so model ids containing "/"
    survive intact."""
    if isinstance(model_string, str) and "/" in model_string:
        provider, model_identifier = model_string.split("/", 1)
        return provider, model_identifier
    return fallback_provider, model_string


class PiOutputEventNormalizer:
    """Maps Pi's own event stream to the runner's three internal chunk shapes.

    turn_end -> an ``assistant`` chunk (the turn's text) + a ``stream_event``
    message_delta carrying usage; agent_end -> the terminal ``result`` chunk with
    the accumulated final text, usage, and the minted session id. The session id
    is remembered from the first ``session`` event.
    """

    def __init__(self) -> None:
        self._minted_session_id = None
        self._latest_final_text = ""
        self._latest_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        # Session-cumulative token BASELINE (pi's raw key space: input/output/
        # cacheRead/cacheWrite), so an abort-time get_session_stats response can
        # be reported as a DELTA — only the tokens this run's un-finished turn
        # actually spent, never the whole session again (a resumed session's
        # stats include earlier runs, and completed turns already reported
        # their usage on turn_end). None until the first stats response.
        self._session_cumulative_token_baseline = None
        self._abort_response_seen = False
        # Generic post-abort contract with the core: declaring this attribute
        # (False) asks the core for a bounded read of remaining output after an
        # abort, until it flips True (set when the post-abort stats arrive).
        self.final_usage_after_abort_reported = False

    @staticmethod
    def _add_pi_usage_onto_baseline(baseline: dict, pi_usage: dict) -> None:
        for pi_key in ("input", "output", "cacheRead", "cacheWrite"):
            value = (pi_usage or {}).get(pi_key)
            if isinstance(value, (int, float)):
                baseline[pi_key] = baseline.get(pi_key, 0) + int(value)

    @staticmethod
    def _extract_assistant_text(message) -> str:
        if not isinstance(message, dict):
            return ""
        text_parts = []
        for block in message.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    @staticmethod
    def _map_usage(pi_usage) -> dict:
        pi_usage = pi_usage if isinstance(pi_usage, dict) else {}
        return {
            "input_tokens": int(pi_usage.get("input") or 0),
            "output_tokens": int(pi_usage.get("output") or 0),
            "cache_creation_input_tokens": int(pi_usage.get("cacheWrite") or 0),
            "cache_read_input_tokens": int(pi_usage.get("cacheRead") or 0),
        }

    def normalize(self, raw_chunk: dict) -> List[dict]:
        if not isinstance(raw_chunk, dict):
            return []
        event_type = raw_chunk.get("type")

        # RPC command acknowledgements (``{"type":"response","command":..,"success":..}``)
        # are protocol bookkeeping for our prompt/abort/steer commands, NOT agent events
        # — drop them so they never reach the internal chunk stream or the live log's
        # event mapping. (json mode never emitted these.)
        if event_type == "response":
            # get_state's response is the ONLY place RPC mode reports the
            # session id (docs/rpc.md — there is no spontaneous session event).
            # Surface it as the uniform session chunk (id under "session_id")
            # so the core captures it like any other harness's announcement.
            if raw_chunk.get("command") == "get_state":
                state_data = raw_chunk.get("data") or {}
                state_session_id = state_data.get("sessionId")
                if isinstance(state_session_id, str) and state_session_id:
                    self._minted_session_id = state_session_id
                    return [{"type": "session", "session_id": state_session_id}]
            if raw_chunk.get("command") == "abort":
                # The NEXT get_session_stats response is the post-abort final
                # usage read (requested by request_abort right after the abort).
                self._abort_response_seen = True
                return []
            if raw_chunk.get("command") == "get_session_stats":
                stats_tokens = (raw_chunk.get("data") or {}).get("tokens")
                if not isinstance(stats_tokens, dict):
                    if self._abort_response_seen:
                        self.final_usage_after_abort_reported = True
                    return []
                if self._session_cumulative_token_baseline is None:
                    # First stats response of this run = the baseline (a fresh
                    # session reports 0s; a RESUMED session reports the tokens
                    # earlier runs already accounted for — never re-report them).
                    self._session_cumulative_token_baseline = {
                        pi_key: int(stats_tokens.get(pi_key) or 0)
                        for pi_key in ("input", "output", "cacheRead", "cacheWrite")
                    }
                    if self._abort_response_seen:
                        self.final_usage_after_abort_reported = True
                    return []
                unreported_usage_delta = {
                    pi_key: max(
                        0,
                        int(stats_tokens.get(pi_key) or 0)
                        - self._session_cumulative_token_baseline.get(pi_key, 0),
                    )
                    for pi_key in ("input", "output", "cacheRead", "cacheWrite")
                }
                self._session_cumulative_token_baseline = {
                    pi_key: int(stats_tokens.get(pi_key) or 0)
                    for pi_key in ("input", "output", "cacheRead", "cacheWrite")
                }
                if self._abort_response_seen:
                    self.final_usage_after_abort_reported = True
                if any(unreported_usage_delta.values()):
                    return [{
                        "type": "stream_event",
                        "event": {
                            "type": "message_delta",
                            "usage": self._map_usage(unreported_usage_delta),
                        },
                    }]
                return []
            return []

        if event_type == "session":
            session_id = raw_chunk.get("id")
            if isinstance(session_id, str) and session_id:
                self._minted_session_id = session_id
                # Emit the session announcement in the UNIFORM chunk shape
                # (session id under "session_id", like claude's system-init
                # event) so the core captures it the same way for every
                # harness — including runs that never reach agent_end
                # (aborted / operator-ended runs).
                return [{"type": "session", "session_id": session_id}]
            return []

        if event_type == "turn_end":
            message = raw_chunk.get("message") or {}
            turn_text = self._extract_assistant_text(message)
            if turn_text:
                self._latest_final_text = turn_text
            self._latest_usage = self._map_usage(message.get("usage"))
            # A COMPLETED turn reports its own usage (the message_delta below);
            # advance the session baseline by the same amount so an abort-time
            # stats read never re-reports what this turn already reported.
            if self._session_cumulative_token_baseline is not None:
                self._add_pi_usage_onto_baseline(
                    self._session_cumulative_token_baseline, message.get("usage")
                )
            emitted: List[dict] = []
            if turn_text:
                emitted.append(
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": turn_text}]}}
                )
            emitted.append(
                {"type": "stream_event", "event": {"type": "message_delta", "usage": dict(self._latest_usage)}}
            )
            return emitted

        if event_type == "agent_end":
            return [
                {
                    "type": "result",
                    "result": self._latest_final_text,
                    "usage": dict(self._latest_usage),
                    "session_id": self._minted_session_id,
                    "harness": HARNESS_ID_PI,
                }
            ]

        # Everything else (agent_start, turn_start, message_start/update/end,
        # tool_execution_*) is informational; keep it in the live log as-is.
        return [raw_chunk]


class PiHarnessIntegration:
    harness_id = HARNESS_ID_PI
    capabilities = HarnessCapabilities(
        supports_operator_permission_mode=False,
        supports_caller_chosen_session_id=False,
        supports_session_prime_and_fork=False,
        supports_multimodal_input=False,
        # RPC mode keeps stdin OPEN after the prompt, so mid-run command injection
        # (prompt / steer / follow_up / abort) is possible — the graceful in-process
        # abort path depends on this.
        supports_mid_run_command_injection=True,
        # Pi is always full-auto and has no operator permission channel: a
        # permission_mode (e.g. from a collaborative task) is silently ignored
        # rather than rejected, so such a task still dispatches.
        ignores_unsupported_permission_mode=True,
    )

    def build_launch_command(self, run_request) -> List[str]:
        pi_command = os.environ.get(PI_COMMAND_ENVIRONMENT_VARIABLE, "pi")
        fallback_provider = os.environ.get(
            PI_PROVIDER_ENVIRONMENT_VARIABLE, DEFAULT_PI_PROVIDER
        )
        provider, model_identifier = split_model_string_into_provider_and_model(
            run_request.model, fallback_provider
        )
        session_directory = os.environ.get(
            PI_SESSION_DIRECTORY_ENVIRONMENT_VARIABLE
        ) or os.path.join(os.fspath(run_request.workspace_directory), "pi_sessions")

        argv = [
            pi_command,
            # --mode rpc is the persistent, bidirectional streaming form: the process
            # STAYS ALIVE with stdin OPEN, reading JSONL commands (prompt / steer /
            # follow_up / abort) one per line and emitting the SAME LF-delimited event
            # stream json mode does (message_update text deltas, tool events, turn_end,
            # agent_end) live. This lets the runner send a graceful {"type":"abort"}
            # that cancels the upstream llama.cpp generation without an OS-kill, and
            # continue a session's follow-up turns in-process (no per-turn re-prefill).
            "--mode",
            "rpc",
            "--provider",
            provider,
            "--session-dir",
            session_directory,
            # Pi is driven bare in the open VM: no ambient context files, skills,
            # prompt templates, themes, or auto-discovered extensions.
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-extensions",
        ]
        if model_identifier:
            argv += ["--model", model_identifier]
        # Resume an existing Pi session by its minted id; a fresh run omits
        # --session and lets Pi mint one (captured from the session event).
        if run_request.session_id and run_request.resume_session:
            argv += ["--session", run_request.session_id]
        # Explicitly loaded extensions (e.g. the web_search-via-SearXNG tool).
        extension_paths_raw = os.environ.get(PI_EXTENSION_PATHS_ENVIRONMENT_VARIABLE, "")
        for extension_path in extension_paths_raw.split(os.pathsep):
            if extension_path.strip():
                argv += ["-e", extension_path.strip()]
        # Per-session tool allowlist: only when the session assigns an exclusive
        # whitelist. Runner-provided MCP names (e.g. unharness-api) are HELD OUT of
        # the harness tool flag — they are not native Pi tools; the runner mediates
        # them. So --tools carries only the BUILT-IN names: an empty built-in set
        # => --tools "" (no native tools, the API-only posture for ingest tasks);
        # a populated set => exactly those native tools.
        if (
            run_request.restrict_to_assigned_tools_as_whitelist
            and run_request.assigned_tool_list is not None
        ):
            builtin_tool_names, _runner_mcp_names = (
                split_assigned_tools_into_builtins_and_runner_mcps(
                    run_request.assigned_tool_list
                )
            )
            argv += ["--tools", ",".join(builtin_tool_names)]
        # Session-start steering appended to Pi's system prompt (e.g. the pointer
        # to the VM environment docs when bash is enabled).
        if run_request.append_system_prompt_text:
            argv += ["--append-system-prompt", run_request.append_system_prompt_text]
        argv.extend(run_request.extra_cli_flags)
        return argv

    def deliver_prompt(
        self,
        *,
        process,
        input_content,
        run_request,
        write_stream_json_message,
        permission_prompt_enabled,
    ) -> PromptDeliveryOutcome:
        # RPC mode: send the prompt as a JSONL command and LEAVE stdin OPEN so the
        # runner can inject follow-up turns and a graceful {"type":"abort"} later.
        prompt_text = "\n\n".join(
            block.text for block in input_content if isinstance(block, TextBlock)
        )
        write_pi_rpc_command_line(
            process, {"id": _fresh_rpc_command_id(), "type": "prompt", "message": prompt_text}
        )
        # RPC mode never announces the session id spontaneously (verified from
        # docs/rpc.md): it is only available via get_state's response
        # (data.sessionId). Ask for it right after the prompt so the normalizer
        # can report the minted/resumed session id early — even for runs later
        # aborted mid-turn (a paused run must stay resumable).
        write_pi_rpc_command_line(
            process, {"id": _fresh_rpc_command_id(), "type": "get_state"}
        )
        # Baseline token read: the first get_session_stats response anchors the
        # session-cumulative baseline (a RESUMED session starts non-zero), so a
        # later abort-time stats read reports only THIS run's un-reported tokens.
        write_pi_rpc_command_line(
            process, {"id": _fresh_rpc_command_id(), "type": "get_session_stats"}
        )
        return PromptDeliveryOutcome(process_stdin_remains_open=True)

    def create_output_event_normalizer(self) -> PiOutputEventNormalizer:
        return PiOutputEventNormalizer()

    def deliver_followup_turn(
        self,
        *,
        current_process,
        message_text,
        session_id,
        run_request,
        launch_harness_subprocess,
    ):
        # RPC mode: the SAME process is still alive with stdin open, so continue the
        # session IN-PROCESS by sending another prompt command — NO fresh --session
        # spawn, NO re-prefill of the growing context. The turn just ended (the tool
        # loop serves a request after agent_end), so the agent is idle and no
        # streamingBehavior is required. Return the SAME process for the core to keep
        # reading turn output from.
        write_pi_rpc_command_line(
            current_process,
            {"id": _fresh_rpc_command_id(), "type": "prompt", "message": message_text},
        )
        return current_process

    def deliver_injected_command(
        self, *, process, command_text, write_stream_json_message
    ) -> None:
        """Translate a GENERIC operator "send command" (inject a message into the
        running agent) into pi's RPC form: a ``steer`` command, which pi queues and
        delivers after the current turn's tool calls, before the next LLM call. The
        core stays harness-agnostic — it hands down the raw command_text; only this
        adapter knows pi's wire shape."""
        write_pi_rpc_command_line(
            process, {"type": "steer", "message": command_text}
        )

    def request_abort(self, *, process, write_stream_json_message) -> None:
        """Send the RPC ``{"type":"abort"}`` command to gracefully cancel the current
        operation (verified live 2026-07-20: llama.cpp stops generating ~0.03s later)
        WITHOUT killing the process/session — the runner's control-channel end path
        calls this before it terminates the one-shot task's process. Best-effort.

        Also asks for ``get_session_stats`` right after: the aborted partial
        turn never reaches ``turn_end`` (the only usage-bearing event), so this
        final stats read is the ONLY way the paused run's spent tokens get
        reported. The normalizer emits the un-reported delta and flips
        ``final_usage_after_abort_reported`` so the core's bounded post-abort
        read knows when to stop."""
        write_pi_rpc_command_line(process, {"type": "abort"})
        write_pi_rpc_command_line(process, {"type": "get_session_stats"})


register_harness_integration(PiHarnessIntegration())

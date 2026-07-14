"""The Pi CLI harness integration — a fully isolated implementation of
``HarnessIntegration`` on Pi's own terms, NOT modeled on any other harness.

Pi (``@earendil-works/pi-coding-agent``) is driven non-interactively in JSON
event mode: ``pi --print --mode json … `` streams LF-delimited JSON events and
exits on its own. Verified behavior (pi 0.74.2):
  * The prompt is read from STDIN in --print mode (so the runner delivers it over
    stdin, its standard path; nothing is placed on the process argv).
  * A fresh run (no --session) makes Pi MINT its own session id, reported on the
    first ``{"type":"session","id":...}`` event; a later turn resumes it with
    ``--session <id>``. There is no caller-chosen create id.
  * Final assistant text + token usage per turn arrive on ``turn_end``; the run's
    terminal marker is ``agent_end``.

Pi is a reduced-capability harness: no operator permission protocol, mints its
own session ids, no prime/fork reuse, text input only, stdin closed to start.

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

import dataclasses
import os
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

        if event_type == "session":
            session_id = raw_chunk.get("id")
            if isinstance(session_id, str) and session_id:
                self._minted_session_id = session_id
            return []

        if event_type == "turn_end":
            message = raw_chunk.get("message") or {}
            turn_text = self._extract_assistant_text(message)
            if turn_text:
                self._latest_final_text = turn_text
            self._latest_usage = self._map_usage(message.get("usage"))
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
        supports_mid_run_command_injection=False,
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
            "--print",
            "--mode",
            "json",
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
        # Pi reads the prompt as plain text from stdin in --print mode and starts
        # on EOF; write the text blocks and close stdin (no mid-run injection).
        prompt_text = "\n\n".join(
            block.text for block in input_content if isinstance(block, TextBlock)
        )
        if process.stdin is not None:
            try:
                process.stdin.write(prompt_text)
                process.stdin.flush()
                process.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
        return PromptDeliveryOutcome(process_stdin_remains_open=False)

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
        # Pi runs ONE process per turn and has already exited by now. Continue the
        # SAME Pi session as a fresh `pi --session <id>` run, delivering the
        # follow-up message on the new process's stdin (Pi reads the prompt from
        # stdin in --print mode). Return the NEW process for the core to read.
        try:
            current_process.wait(timeout=30)
        except Exception:  # noqa: BLE001 — never block the loop on a stuck exit
            pass
        resume_run_request = dataclasses.replace(
            run_request, session_id=session_id, resume_session=True
        )
        followup_argv = self.build_launch_command(resume_run_request)
        followup_process = launch_harness_subprocess(followup_argv)
        if followup_process.stdin is not None:
            try:
                followup_process.stdin.write(message_text)
                followup_process.stdin.flush()
                followup_process.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
        return followup_process


register_harness_integration(PiHarnessIntegration())

"""The Claude Code CLI harness integration — fully isolated implementation of
``HarnessIntegration``, owning its own argv construction (base streaming argv +
the priming/fork argvs used by claude's session-reuse paths). Claude is the rich
harness: operator permission protocol, caller-chosen/resumable session ids,
prime-and-fork session reuse, multimodal input, and mid-run command injection
over an open stdin. Its stdout already emits the runner's internal chunk shapes,
so its output normalizer is the identity.
"""

from __future__ import annotations

import json
from typing import List

from .content import build_injected_user_message, build_user_message
from .harness_integration import (
    HarnessCapabilities,
    IdentityOutputEventNormalizer,
    PromptDeliveryOutcome,
    register_harness_integration,
)
from .harness_tool_name_translation import (
    translate_generic_tool_names_to_harness_specific,
)
from .request import RunRequest
from .runner_provided_mcp_registry import (
    split_assigned_tools_into_builtins_and_runner_mcps,
)

HARNESS_ID_CLAUDE_CLI = "claude_cli"


def build_base_claude_argv(run_request: RunRequest) -> "list[str]":
    """The verified streaming claude argv (no prompt positional).

    stream-json out with partial messages + stream-json in (so the prompt and
    any send_command can be injected over stdin) + --verbose (required for
    stream-json output in -p mode).
    """
    argv = [
        run_request.claude_command,
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--input-format",
        "stream-json",
        "--verbose",
    ]
    # An explicit permission posture WINS over full bypass: a collaborative /
    # manual task launches at --permission-mode <mode> and is NOT given
    # --dangerously-skip-permissions, so the agent is not fully unattended.
    if run_request.permission_mode:
        argv[2:2] = ["--permission-mode", run_request.permission_mode]
        # Live tool-permission escalation: with a permission posture set, drive the
        # CLI's can_use_tool control protocol over stdio so tools needing approval
        # emit a request the runner can answer (allow/deny) — verified flag on the
        # claude CLI (real but not shown in --help). The runner sends the
        # initialize handshake + control_response decisions.
        argv += ["--permission-prompt-tool", "stdio"]
    elif run_request.dangerously_skip_permissions:
        argv.insert(2, "--dangerously-skip-permissions")
    # Compose ONE inline --settings argument (raw-1252): the caller's claude settings
    # overrides (effort/thinking, etc.) MERGED with the permission-mode default posture.
    # Inline --settings applies on the host AND a remote/VM claude regardless of cwd and
    # OUTRANKS the environment's settings.json (closing the gap where a per-workspace
    # settings.local.json is not read by a remote claude). Emitted only when non-empty.
    combined_claude_settings = dict(run_request.claude_settings_overrides or {})
    if run_request.permission_mode:
        # Pin the session's default permission posture so a manual/collaborative run
        # actually GATES — otherwise a host/VM standing ``defaultMode:
        # bypassPermissions`` leaks through and auto-allows tools (verified on the
        # sandbox VM), defeating the prompt.
        permissions_block = dict(combined_claude_settings.get("permissions") or {})
        permissions_block["defaultMode"] = run_request.permission_mode
        combined_claude_settings["permissions"] = permissions_block
    if combined_claude_settings:
        argv += ["--settings", json.dumps(combined_claude_settings, sort_keys=True)]
    # Caller session-start steering (raw-1255): append text to the system prompt via
    # the official --append-system-prompt flag. Part of the (stable) system prompt, so
    # it stays in the cached prefix shared by warm forks. Emitted only when non-empty.
    if run_request.append_system_prompt_text:
        argv += ["--append-system-prompt", run_request.append_system_prompt_text]
    if run_request.model:
        argv[2:2] = ["--model", run_request.model]
    # Explicit session id so the session is resumable across turns (resume-on-
    # reply, collaborative tasks). resume_session => CONTINUE the existing
    # session; otherwise CREATE it with this id. (The prime/fork reuse path uses
    # its own argv builders and never sets run_request.session_id, so no clash.)
    if run_request.session_id:
        if run_request.resume_session:
            argv += ["--resume", run_request.session_id]
        else:
            argv += ["--session-id", run_request.session_id]
    # Per-session tool restriction. When the session's assigned tool list is enforced
    # as an exclusive whitelist, restrict the harness to ONLY the assigned BUILT-IN
    # tools via --tools (the runner-provided MCP names are held out — they are enabled
    # through the runner's MCP-client layer, not the harness tool flag; and --tools
    # governs built-ins only, per the CLI docs). Verified on the installed CLI:
    # `--tools ""` disables all built-ins, `--tools "Bash,Edit,Read"` limits to those.
    # (This is distinct from --allowedTools, which only pre-approves.) Names are
    # comma-joined into one argument so the flag's greedy nargs cannot swallow later
    # flags. Not restricting => emit no --tools (harness default toolset).
    if (
        run_request.restrict_to_assigned_tools_as_whitelist
        and run_request.assigned_tool_list is not None
    ):
        generic_builtin_tool_names, _runner_mcp_names = (
            split_assigned_tools_into_builtins_and_runner_mcps(
                run_request.assigned_tool_list
            )
        )
        # The assigned names are GENERIC (Unharness-level); convert to claude-specific
        # names before emitting --tools (the runner owns this harness translation).
        claude_specific_tool_names = translate_generic_tool_names_to_harness_specific(
            generic_builtin_tool_names, HARNESS_ID_CLAUDE_CLI
        )
        argv += ["--tools", ",".join(claude_specific_tool_names)]
        # --tools governs BUILT-INS only (CLI docs): account-level MCP servers
        # (e.g. the login's claude.ai connectors — Gmail/Drive/Calendar
        # authenticate stubs) still load and leak into a whitelisted session
        # (observed live: a session accidentally called Gmail authenticate).
        # Per the docs, --strict-mcp-config WITHOUT --mcp-config loads NO MCP
        # servers — a whitelisted session gets exactly its assigned tools.
        argv += ["--strict-mcp-config"]
    argv.extend(run_request.extra_cli_flags)
    return argv


def build_priming_claude_argv(
    run_request: RunRequest, primed_session_id: str, chunk_text: str
) -> "list[str]":
    """Argv for a PRIMING run: a SIMPLE, self-completing ``claude -p`` invocation
    that creates a NEW session with a caller-chosen id (``--session-id``) and
    ingests the chunk as a positional prompt.

    VERIFIED against real claude: the streaming base argv (stream-json in/out,
    chunk over stdin) does NOT complete a priming session — no result is ever
    produced, so priming always failed and reuse fell back to inline. A plain
    completing call DOES persist a forkable primed session:

        claude [--model M] [--dangerously-skip-permissions] \\
               --session-id <primed_sid> -p "<chunk text>"

    NO --output-format/--input-format/--include-partial-messages/--verbose and
    NO stdin: the prompt is the positional arg, and claude exits 0 on its own.
    Later task runs fork this session (the chunk is then a cache read).
    """
    argv = [run_request.claude_command]
    if run_request.model:
        argv += ["--model", run_request.model]
    if run_request.dangerously_skip_permissions:
        argv += ["--dangerously-skip-permissions"]
    argv += ["--session-id", primed_session_id, "-p", chunk_text]
    return argv


def build_fork_claude_argv(
    run_request: RunRequest, primed_session_id: str
) -> "list[str]":
    """Argv for a TASK run that FORKS from an already-primed session.

    ``claude --resume <primed> --fork-session`` creates a new session that inherits
    the primed session's history; the primed session is untouched and reusable. The
    NEW forked session id is MINTED BY CLAUDE and reported on the run's result event —
    we do NOT pass ``--session-id`` here. Combining ``--session-id`` with
    ``--fork-session`` is undocumented and behaved inconsistently live (claude sometimes
    left the passed id a near-empty stub and wrote the work to its own minted id,
    raw-1224..1229). So the caller CAPTURES claude's minted fork id, exactly as it
    captures opencode's minted id. The per-task input is delivered over stdin as usual.
    """
    argv = build_base_claude_argv(run_request)
    argv += ["--resume", primed_session_id, "--fork-session"]
    return argv


class ClaudeHarnessIntegration:
    harness_id = HARNESS_ID_CLAUDE_CLI
    capabilities = HarnessCapabilities(
        supports_operator_permission_mode=True,
        supports_caller_chosen_session_id=True,
        supports_session_prime_and_fork=True,
        supports_multimodal_input=True,
        supports_mid_run_command_injection=True,
    )

    def build_launch_command(self, run_request) -> List[str]:
        return build_base_claude_argv(run_request)

    def deliver_prompt(
        self,
        *,
        process,
        input_content,
        run_request,
        write_stream_json_message,
        permission_prompt_enabled,
    ) -> PromptDeliveryOutcome:
        # Deliver the multimodal prompt as a stdin stream-json user message; stdin
        # stays OPEN afterwards so mid-run send_command injection still works.
        write_stream_json_message(build_user_message(input_content))
        return PromptDeliveryOutcome(process_stdin_remains_open=True)

    def create_output_event_normalizer(self) -> IdentityOutputEventNormalizer:
        return IdentityOutputEventNormalizer()

    def deliver_injected_command(
        self, *, process, command_text, write_stream_json_message
    ) -> None:
        """Translate a GENERIC operator "send command" into claude's stream-json form:
        an injected user message on the running process's open stdin. The core hands
        down the raw command_text; the claude-specific message shape lives HERE."""
        write_stream_json_message(build_injected_user_message(command_text))

    def deliver_followup_turn(
        self,
        *,
        current_process,
        message_text,
        session_id,
        run_request,
        launch_harness_subprocess,
    ):
        # claude runs ONE long-lived streaming process: inject the follow-up as a
        # stream-json user message on its still-open stdin; the same process keeps
        # streaming the next turn.
        if current_process.stdin is not None:
            try:
                current_process.stdin.write(
                    json.dumps(build_injected_user_message(message_text)) + "\n"
                )
                current_process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass
        return current_process


register_harness_integration(ClaudeHarnessIntegration())

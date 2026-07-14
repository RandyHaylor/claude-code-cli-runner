"""The common harness interface: the single, clean surface the streaming runner
core uses to drive ANY agent-CLI harness, so the core holds ZERO
``if harness == ...`` conditionals.

Each concrete harness (claude, opencode, pi, ...) is a fully ISOLATED
implementation of ``HarnessIntegration`` living in its own module; they do not
share integration logic with each other (duplication is preferred over coupling
here — the harnesses are genuinely different CLIs with different feature sets).

The core asks a harness integration for exactly these things:
  * ``capabilities`` — a declaration of what the harness supports, so the core
    (and request validation) can gate behavior generically instead of naming
    harnesses.
  * ``build_launch_command(run_request)`` — the argv that launches ONE streaming
    run of this harness (no execution-location wrapping; that stays in the
    transports layer).
  * ``deliver_prompt(...)`` — write the run's prompt to the launched process on
    the harness's own terms, and report whether the process stdin stays open
    afterwards (i.e. whether mid-run command injection is possible).
  * ``create_output_event_normalizer()`` — a fresh per-run object that maps the
    harness's raw stdout events into the runner's internal chunk shapes.

The runner's internal chunk shapes every normalizer must emit:
  1. ``{"type": "assistant", "message": {"content": [{"type": "text", ...}]}}``
  2. ``{"type": "stream_event", "event": {"type": "message_delta", "usage": {...}}}``
  3. ``{"type": "result", "result": "<final text>", "usage": {...}}`` (terminal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol


@dataclass(frozen=True)
class HarnessCapabilities:
    """What a harness supports. The core and request validation read THESE flags
    instead of naming specific harnesses, so adding a harness never means editing
    a scattered set of ``if harness == ...`` checks."""

    # Honors an operator-gated permission posture (permission_mode + a stdio
    # can-use-tool escalation protocol). When False, permission_mode is rejected.
    supports_operator_permission_mode: bool
    # Can CREATE a session with a caller-chosen id. When False, the harness mints
    # its own id (a caller-chosen create id is rejected; resume-by-id may still work).
    supports_caller_chosen_session_id: bool
    # Supports the prime-once / fork-per-task session-reuse and warm-base-fork paths.
    supports_session_prime_and_fork: bool
    # Accepts non-text (image/document) input blocks.
    supports_multimodal_input: bool
    # Leaves stdin open after the prompt so mid-run operator command injection works.
    supports_mid_run_command_injection: bool
    # When a permission_mode is set but this harness has no operator permission
    # protocol: True => SILENTLY IGNORE it (harness is always full-auto — e.g. pi);
    # False => reject it (refuse to run ungated — e.g. opencode). Only consulted when
    # supports_operator_permission_mode is False. Defaults False (reject).
    ignores_unsupported_permission_mode: bool = False


@dataclass(frozen=True)
class PromptDeliveryOutcome:
    """What ``deliver_prompt`` reports back to the core after writing the prompt."""

    # True if the process stdin remains OPEN (mid-run injection still possible);
    # False if the harness required stdin to be closed to start work.
    process_stdin_remains_open: bool


class HarnessOutputEventNormalizer(Protocol):
    """A per-run, possibly-stateful object that turns each raw harness stdout
    chunk into zero or more runner-internal chunks."""

    def normalize(self, raw_chunk: dict) -> List[dict]:
        ...


class IdentityOutputEventNormalizer:
    """For a harness whose stdout already IS the runner's internal chunk shape:
    pass every chunk through unchanged."""

    def normalize(self, raw_chunk: dict) -> List[dict]:
        return [raw_chunk]


class HarnessIntegration(Protocol):
    """The contract every harness implementation satisfies. Concrete
    implementations live in per-harness modules and are registered below."""

    harness_id: str
    capabilities: HarnessCapabilities

    def build_launch_command(self, run_request) -> List[str]:
        ...

    def deliver_prompt(
        self,
        *,
        process,
        input_content,
        run_request,
        write_stream_json_message: Callable[[dict], None],
        permission_prompt_enabled: bool,
    ) -> PromptDeliveryOutcome:
        ...

    def create_output_event_normalizer(self) -> HarnessOutputEventNormalizer:
        ...

    def deliver_followup_turn(
        self,
        *,
        current_process,
        message_text,
        session_id,
        run_request,
        launch_harness_subprocess,
    ):
        """Continue the run's session with an additional user message (the shared
        surface the core's runner-mediated tool loop calls after it serves a tool
        request). Returns the process the core should keep reading turn output from.

        Each harness continues its own way: a harness with a long-lived streaming
        process writes the message to that process's open stdin and returns the
        SAME process; a harness that runs one process per turn waits for the
        current process to exit and launches a fresh resume-the-session process
        (via ``launch_harness_subprocess``), returning the NEW process. The core
        tool loop neither knows nor cares which — it just reads the returned
        process's output.

        ``launch_harness_subprocess(argv) -> subprocess.Popen`` launches a process
        with the run's cwd + environment (provided by the core so a per-turn
        harness reuses the exact launch posture).
        """
        ...


# --- registry ---------------------------------------------------------------
# harness id -> the single integration instance. Integrations register themselves
# at import time (see each per-harness module). The core resolves via
# ``get_harness_integration``.

_HARNESS_INTEGRATION_BY_ID: "dict[str, HarnessIntegration]" = {}


def register_harness_integration(integration: "HarnessIntegration") -> None:
    _HARNESS_INTEGRATION_BY_ID[integration.harness_id] = integration


def get_harness_integration(harness_id: str) -> "HarnessIntegration":
    try:
        return _HARNESS_INTEGRATION_BY_ID[harness_id]
    except KeyError:
        raise ValueError(
            "unknown harness %r; registered harnesses: %s"
            % (harness_id, ", ".join(sorted(_HARNESS_INTEGRATION_BY_ID)))
        )


def known_harness_ids() -> "tuple[str, ...]":
    return tuple(sorted(_HARNESS_INTEGRATION_BY_ID))


def get_harness_integration_or_none(harness_id: str) -> "Optional[HarnessIntegration]":
    return _HARNESS_INTEGRATION_BY_ID.get(harness_id)

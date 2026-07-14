"""The Claude Code CLI harness integration — fully isolated implementation of
``HarnessIntegration``. Claude is the rich harness: operator permission
protocol, caller-chosen/resumable session ids, prime-and-fork session reuse,
multimodal input, and mid-run command injection over an open stdin.

Its stdout already emits the runner's internal chunk shapes, so its output
normalizer is the identity.
"""

from __future__ import annotations

from typing import List

from .content import build_user_message
from .harness_integration import (
    HarnessCapabilities,
    IdentityOutputEventNormalizer,
    PromptDeliveryOutcome,
    register_harness_integration,
)
from .transports import build_base_claude_argv

HARNESS_ID_CLAUDE_CLI = "claude_cli"


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


register_harness_integration(ClaudeHarnessIntegration())

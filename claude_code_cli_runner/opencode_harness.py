"""The opencode CLI harness integration — fully isolated implementation of
``HarnessIntegration``. opencode is a reduced-capability harness: no operator
permission protocol, it mints its own session ids (no caller-chosen create), no
prime-and-fork reuse, text input only, and it requires stdin CLOSED to start
(so no mid-run command injection).

Its stdout events are opencode-shaped, so its output normalizer wraps the
opencode->internal-chunk translator.
"""

from __future__ import annotations

from typing import List

from .request import TextBlock
from .harness_integration import (
    HarnessCapabilities,
    PromptDeliveryOutcome,
    register_harness_integration,
)
from .opencode_event_translation import OpencodeEventToClaudeChunkTranslator
from .transports import build_base_opencode_argv

HARNESS_ID_OPENCODE_CLI = "opencode_cli"


class _OpencodeOutputEventNormalizer:
    """Per-run wrapper over the stateful opencode->internal-chunk translator."""

    def __init__(self) -> None:
        self._translator = OpencodeEventToClaudeChunkTranslator()

    def normalize(self, raw_chunk: dict) -> List[dict]:
        return self._translator.translate(raw_chunk)


class OpencodeHarnessIntegration:
    harness_id = HARNESS_ID_OPENCODE_CLI
    capabilities = HarnessCapabilities(
        supports_operator_permission_mode=False,
        supports_caller_chosen_session_id=False,
        supports_session_prime_and_fork=False,
        supports_multimodal_input=False,
        supports_mid_run_command_injection=False,
    )

    def build_launch_command(self, run_request) -> List[str]:
        return build_base_opencode_argv(run_request)

    def deliver_prompt(
        self,
        *,
        process,
        input_content,
        run_request,
        write_stream_json_message,
        permission_prompt_enabled,
    ) -> PromptDeliveryOutcome:
        # opencode reads the prompt as PLAIN TEXT from stdin and starts on EOF —
        # write the text blocks and CLOSE stdin. Consequence: mid-run send_command
        # injection is not available (dropped harmlessly on the closed pipe).
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

    def create_output_event_normalizer(self) -> _OpencodeOutputEventNormalizer:
        return _OpencodeOutputEventNormalizer()


register_harness_integration(OpencodeHarnessIntegration())

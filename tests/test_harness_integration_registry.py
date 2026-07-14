"""The common harness interface + registry: registration, capability
declarations, and argv-builder parity with the pre-refactor functions (the
integrations must delegate to exactly the same argv the runner produced before).
"""

from __future__ import annotations

from claude_code_cli_runner.harness_integration import (
    get_harness_integration,
    get_harness_integration_or_none,
    known_harness_ids,
)
from claude_code_cli_runner.claude_harness import (
    HARNESS_ID_CLAUDE_CLI,
    build_base_claude_argv,
)
from claude_code_cli_runner.opencode_harness import (
    HARNESS_ID_OPENCODE_CLI,
    build_base_opencode_argv,
)
from claude_code_cli_runner.request import RunRequest, TextBlock, HARNESS_OPENCODE_CLI


def _make_run_request(**overrides) -> RunRequest:
    base = dict(
        input_content=[TextBlock("hello")],
        workspace_directory="/tmp/harness-iface-test",
        model="some-model",
    )
    base.update(overrides)
    return RunRequest(**base)


def test_both_harnesses_are_registered():
    assert HARNESS_ID_CLAUDE_CLI in known_harness_ids()
    assert HARNESS_ID_OPENCODE_CLI in known_harness_ids()


def test_unknown_harness_id_returns_none_or_raises():
    assert get_harness_integration_or_none("nope") is None
    try:
        get_harness_integration("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown harness id")


def test_capability_declarations_reflect_the_two_harnesses():
    claude = get_harness_integration(HARNESS_ID_CLAUDE_CLI).capabilities
    opencode = get_harness_integration(HARNESS_ID_OPENCODE_CLI).capabilities

    assert claude.supports_operator_permission_mode is True
    assert claude.supports_caller_chosen_session_id is True
    assert claude.supports_session_prime_and_fork is True
    assert claude.supports_multimodal_input is True
    assert claude.supports_mid_run_command_injection is True

    assert opencode.supports_operator_permission_mode is False
    assert opencode.supports_caller_chosen_session_id is False
    assert opencode.supports_session_prime_and_fork is False
    assert opencode.supports_multimodal_input is False
    assert opencode.supports_mid_run_command_injection is False


def test_claude_launch_command_matches_legacy_builder():
    run_request = _make_run_request()
    integration = get_harness_integration(HARNESS_ID_CLAUDE_CLI)
    assert integration.build_launch_command(run_request) == build_base_claude_argv(run_request)


def test_opencode_launch_command_matches_legacy_builder():
    run_request = _make_run_request(
        harness=HARNESS_OPENCODE_CLI,
        dangerously_skip_permissions=True,
    )
    integration = get_harness_integration(HARNESS_ID_OPENCODE_CLI)
    assert integration.build_launch_command(run_request) == build_base_opencode_argv(run_request)


def test_opencode_normalizer_is_fresh_per_call():
    integration = get_harness_integration(HARNESS_ID_OPENCODE_CLI)
    first = integration.create_output_event_normalizer()
    second = integration.create_output_event_normalizer()
    assert first is not second


def test_claude_normalizer_is_identity():
    integration = get_harness_integration(HARNESS_ID_CLAUDE_CLI)
    normalizer = integration.create_output_event_normalizer()
    chunk = {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}
    assert normalizer.normalize(chunk) == [chunk]

"""Loop breaker for a quantized model that repeats the SAME tool call forever
(raw-1785): after more than MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS identical calls in a
row, a corrective message is produced for the core to steer into the agent; a different
call resets the streak. Pi (stdin stays open) supplies the breaker; other harnesses don't.
"""

from claude_code_cli_runner.pi_harness import (
    MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS,
    REPEATED_TOOL_CALL_CORRECTIVE_MESSAGE,
    RepeatedIdenticalToolCallLoopBreaker,
    PiHarnessIntegration,
)

_CALL = ("bash", {"command": "echo '{\"tool\": \"read_my_task_details\"}'"})


def test_no_corrective_up_to_and_including_the_threshold():
    breaker = RepeatedIdenticalToolCallLoopBreaker()
    for _ in range(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS):
        assert breaker.corrective_message_for_observed_tool_call(*_CALL) is None


def test_corrective_fires_once_past_the_threshold():
    breaker = RepeatedIdenticalToolCallLoopBreaker()
    results = [
        breaker.corrective_message_for_observed_tool_call(*_CALL)
        for _ in range(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS + 1)
    ]
    assert results[-1] == REPEATED_TOOL_CALL_CORRECTIVE_MESSAGE
    assert all(r is None for r in results[:-1])
    assert str(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS) in REPEATED_TOOL_CALL_CORRECTIVE_MESSAGE


def test_corrective_re_fires_every_threshold_so_one_ignored_steer_does_not_loop_forever():
    breaker = RepeatedIdenticalToolCallLoopBreaker()
    fired_at = [
        i for i in range(1, 3 * MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS + 2)
        if breaker.corrective_message_for_observed_tool_call(*_CALL)
    ]
    # first past the threshold, then every `threshold` further repeats
    assert fired_at == [
        MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS + 1,
        2 * MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS + 1,
        3 * MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS + 1,
    ]


def test_a_different_call_resets_the_streak():
    breaker = RepeatedIdenticalToolCallLoopBreaker()
    for _ in range(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS):
        breaker.corrective_message_for_observed_tool_call(*_CALL)
    # a different call breaks the streak...
    assert breaker.corrective_message_for_observed_tool_call("read", {"path": "x"}) is None
    # ...so the original call must build up again from scratch
    for _ in range(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS):
        assert breaker.corrective_message_for_observed_tool_call(*_CALL) is None
    assert (
        breaker.corrective_message_for_observed_tool_call(*_CALL)
        == REPEATED_TOOL_CALL_CORRECTIVE_MESSAGE
    )


def test_argument_key_order_does_not_defeat_identity():
    breaker = RepeatedIdenticalToolCallLoopBreaker()
    for _ in range(MAX_IDENTICAL_CONSECUTIVE_TOOL_CALLS):
        breaker.corrective_message_for_observed_tool_call("t", {"a": 1, "b": 2})
    assert (
        breaker.corrective_message_for_observed_tool_call("t", {"b": 2, "a": 1})
        == REPEATED_TOOL_CALL_CORRECTIVE_MESSAGE
    )


def test_pi_integration_supplies_a_breaker_and_supports_injection():
    integration = PiHarnessIntegration()
    assert integration.capabilities.supports_mid_run_command_injection is True
    breaker = integration.create_tool_call_loop_breaker()
    assert isinstance(breaker, RepeatedIdenticalToolCallLoopBreaker)


if __name__ == "__main__":
    import unittest

    unittest.main()

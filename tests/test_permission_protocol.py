"""raw-538/nd-251: the can_use_tool control-protocol building blocks — the
initialize handshake, the permission control_response shape, and the
permission_decision control intent validation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from claude_code_cli_runner.content import (
    build_initialize_control_request,
    build_permission_control_response,
)
from claude_code_cli_runner import live_files


def test_initialize_control_request_shape():
    msg = build_initialize_control_request()
    assert msg["type"] == "control_request"
    assert msg["request"]["subtype"] == "initialize"
    assert msg["request_id"]


def test_permission_control_response_allow_echoes_input():
    resp = build_permission_control_response("req-1", "allow", updated_input={"a": 1})
    assert resp["type"] == "control_response"
    assert resp["response"]["subtype"] == "success"
    assert resp["response"]["request_id"] == "req-1"
    assert resp["response"]["response"] == {"behavior": "allow", "updatedInput": {"a": 1}}


def test_permission_control_response_deny_carries_message():
    resp = build_permission_control_response("req-2", "deny", message="nope")
    inner = resp["response"]["response"]
    assert inner["behavior"] == "deny"
    assert inner["message"] == "nope"


def test_permission_decision_intent_requires_request_id_and_behavior():
    ok = live_files.build_control_intent(
        "permission_decision",
        decision={"request_id": "r1", "behavior": "allow"},
    )
    assert ok["control_intent"] == "permission_decision"
    assert ok["decision"]["behavior"] == "allow"

    with pytest.raises(ValueError):
        live_files.build_control_intent("permission_decision", decision={"behavior": "allow"})
    with pytest.raises(ValueError):
        live_files.build_control_intent(
            "permission_decision", decision={"request_id": "r1", "behavior": "maybe"}
        )
    with pytest.raises(ValueError):
        live_files.build_control_intent("permission_decision")

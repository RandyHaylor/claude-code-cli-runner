"""Architecture test: the live-file names + control intents match the documented
contract EXACTLY, so external readers (e.g. a dashboard) keep working."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import live_files


def test_live_file_names_are_the_contract():
    assert live_files.LIVE_LOG_FILE_NAME == "task_live_log.jsonl"
    assert live_files.CONTROL_CHANNEL_FILE_NAME == "task_control_channel.jsonl"
    assert live_files.RUN_STATUS_FILE_NAME == "task_run_status.json"


def test_control_intents_are_the_contract():
    assert live_files.CONTROL_PAUSE == "pause"
    assert live_files.CONTROL_RESUME == "resume"
    assert live_files.CONTROL_SEND_COMMAND == "send_command"
    assert live_files.CONTROL_END_AND_RETURN == "end_and_return"
    assert live_files.CONTROL_PERMISSION_DECISION == "permission_decision"
    assert set(live_files.KNOWN_CONTROL_INTENTS) == {
        "pause",
        "resume",
        "send_command",
        "end_and_return",
        "permission_decision",
    }


def test_run_states_are_the_contract():
    assert live_files.RUN_STATE_RUNNING == "running"
    assert live_files.RUN_STATE_PAUSED == "paused"
    assert live_files.RUN_STATE_OPERATOR_ENDED == "operator_ended"
    assert live_files.RUN_STATE_AWAITING_PERMISSION == "awaiting_permission"


def test_readme_documents_the_contract():
    readme = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
    )
    with open(readme, encoding="utf-8") as handle:
        text = handle.read()
    for name in ("task_live_log.jsonl", "task_control_channel.jsonl", "task_run_status.json"):
        assert name in text
    for intent in ("pause", "resume", "send_command", "end_and_return"):
        assert intent in text

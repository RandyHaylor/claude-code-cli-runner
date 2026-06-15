"""Control-channel intents: send_command injection and end_and_return."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from claude_code_cli_runner.live_files import (
    CONTROL_END_AND_RETURN,
    CONTROL_SEND_COMMAND,
    append_control_intent,
    control_channel_path,
)
from conftest import stub_build_command


def test_send_command_is_injected_and_echoed(tmp_path):
    # Pre-queue a send_command so it is consumed during the run; the stub echoes
    # injected stdin messages back into stdout, so it shows up in the live log.
    append_control_intent(str(tmp_path), CONTROL_SEND_COMMAND, "do the thing")
    request = RunRequest(
        input_content=[TextBlock(text="start")],
        workspace_directory=str(tmp_path),
    )
    result = run_claude_code_task(request, build_command=stub_build_command)
    with open(result.live_log_path, encoding="utf-8") as handle:
        log_text = handle.read()
    assert "do the thing" in log_text


def test_end_and_return_terminates_run(tmp_path):
    os.environ["STUB_RUN_FOREVER"] = "1"

    def queue_end():
        time.sleep(0.2)
        append_control_intent(str(tmp_path), CONTROL_END_AND_RETURN)

    timer = threading.Thread(target=queue_end)
    timer.start()
    try:
        request = RunRequest(
            input_content=[TextBlock(text="run forever")],
            workspace_directory=str(tmp_path),
        )
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RUN_FOREVER", None)
        timer.join()

    assert result.operator_ended is True
    assert result.run_state == "operator_ended"
    assert os.path.isfile(control_channel_path(str(tmp_path)))

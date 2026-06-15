"""A plain text run against the stub: assistant text + result event + status."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from conftest import stub_build_command


def test_text_run_captures_assistant_text_and_result(tmp_path):
    request = RunRequest(
        input_content=[TextBlock(text="hello stub")],
        workspace_directory=str(tmp_path),
    )
    os.environ["STUB_RESULT_TEXT"] = "the answer is 42"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_RESULT_TEXT", None)

    assert result.assistant_text == "the answer is 42"
    assert result.final_result_event is not None
    assert result.final_result_event.get("type") == "result"
    assert os.path.isfile(result.live_log_path)
    # the prompt we sent was echoed back by the stub via stdin -> stdout
    with open(result.live_log_path, encoding="utf-8") as handle:
        log_text = handle.read()
    assert "injected_echo" in log_text
    assert "hello stub" in log_text

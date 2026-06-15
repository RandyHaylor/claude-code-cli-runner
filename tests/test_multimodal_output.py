"""Multimodal OUTPUT capture: a file the run writes is surfaced as an artifact."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task
from conftest import stub_build_command


def test_produced_artifact_is_surfaced(tmp_path):
    request = RunRequest(
        input_content=[TextBlock(text="make a file")],
        workspace_directory=str(tmp_path),
    )
    os.environ["STUB_WRITE_ARTIFACT"] = "output.txt"
    os.environ["STUB_ARTIFACT_CONTENT"] = "produced by the run"
    try:
        result = run_claude_code_task(request, build_command=stub_build_command)
    finally:
        os.environ.pop("STUB_WRITE_ARTIFACT", None)
        os.environ.pop("STUB_ARTIFACT_CONTENT", None)

    produced_names = [os.path.basename(path) for path in result.produced_artifacts]
    assert "output.txt" in produced_names
    # live-window files are NOT counted as produced artifacts
    assert "task_live_log.jsonl" not in produced_names
    assert "task_run_status.json" not in produced_names

#!/usr/bin/env python3
"""Offline example: a local text run against the STUB claude (never the real CLI).

    python3 examples/run_local_text.py

Points the runner at tests/stub_streaming_claude.py via an injected
build_command, so it runs with no real `claude` installed.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from claude_code_cli_runner import RunRequest, TextBlock, run_claude_code_task

STUB = os.path.join(ROOT, "tests", "stub_streaming_claude.py")


def stub_build_command(run_request):
    return [sys.executable, STUB]


def main():
    workspace = tempfile.mkdtemp(prefix="ccr-example-")
    os.environ["STUB_RESULT_TEXT"] = "Hello from the stub claude!"
    request = RunRequest(
        input_content=[TextBlock(text="Say hello.")],
        workspace_directory=workspace,
    )
    result = run_claude_code_task(request, build_command=stub_build_command)
    print("assistant_text:", result.assistant_text)
    print("run_state:", result.run_state)
    print("live_log_path:", result.live_log_path)
    print("produced_artifacts:", result.produced_artifacts)


if __name__ == "__main__":
    main()

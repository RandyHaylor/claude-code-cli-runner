#!/usr/bin/env python3
"""Offline example: a multimodal (text + image) run against the STUB claude.

    python3 examples/run_multimodal.py

Shows how an image block is added as DATA (an ImageBlock in input_content),
never as new code, and serialized into the stream-json user message.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from claude_code_cli_runner import ImageBlock, RunRequest, TextBlock, run_claude_code_task

STUB = os.path.join(ROOT, "tests", "stub_streaming_claude.py")


def stub_build_command(run_request):
    return [sys.executable, STUB]


def main():
    workspace = tempfile.mkdtemp(prefix="ccr-example-")
    image_path = os.path.join(workspace, "pic.png")
    with open(image_path, "wb") as handle:
        handle.write(b"\x89PNG-not-a-real-image")

    os.environ["STUB_WRITE_ARTIFACT"] = "summary.md"
    request = RunRequest(
        input_content=[
            TextBlock(text="Describe this image and write summary.md."),
            ImageBlock(mime_type="image/png", path=image_path),
        ],
        workspace_directory=workspace,
    )
    result = run_claude_code_task(request, build_command=stub_build_command)
    print("assistant_text:", result.assistant_text)
    print("produced_artifacts:", result.produced_artifacts)


if __name__ == "__main__":
    main()

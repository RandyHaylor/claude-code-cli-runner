"""Serialize input content blocks into the `claude -p` stream-json user message.

Multimodal IN: the runner delivers the prompt as a stdin stream-json user
message whose ``content`` is an array of Anthropic-style content blocks. This
module turns the generic, transport-agnostic :mod:`request` blocks into that
array. Adding a modality is DATA (a new block dataclass + a serializer branch
keyed off its ``block_type``), never a new code path in the runner.
"""

from __future__ import annotations

import base64
import os

from .request import (
    BLOCK_TYPE_DOCUMENT,
    BLOCK_TYPE_IMAGE,
    BLOCK_TYPE_TEXT,
    DocumentBlock,
    ImageBlock,
    TextBlock,
)


def _load_base64(block) -> str:
    """Return the base64 payload for an image/document block, reading+encoding
    the file at ``path`` when no inline ``data_base64`` was supplied."""
    if block.data_base64 is not None:
        return block.data_base64
    if block.path is None:
        raise ValueError(
            "%s block needs either 'path' or 'data_base64'" % block.block_type
        )
    with open(block.path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def serialize_block(block) -> dict:
    """Serialize one content block to its stream-json content-array entry."""
    if isinstance(block, TextBlock) or getattr(block, "block_type", None) == BLOCK_TYPE_TEXT:
        return {"type": "text", "text": block.text}

    if isinstance(block, ImageBlock) or getattr(block, "block_type", None) == BLOCK_TYPE_IMAGE:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.mime_type,
                "data": _load_base64(block),
            },
        }

    if isinstance(block, DocumentBlock) or getattr(block, "block_type", None) == BLOCK_TYPE_DOCUMENT:
        entry = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": block.mime_type,
                "data": _load_base64(block),
            },
        }
        if getattr(block, "name", None):
            entry["title"] = block.name
        return entry

    raise ValueError("unsupported content block: %r" % (block,))


def build_user_message(input_content: list) -> dict:
    """Build the stream-json ``user`` message that carries the full multimodal
    input as a content array. This is written to the process stdin to deliver
    the prompt (real claude in --input-format stream-json mode waits for it)."""
    content = [serialize_block(block) for block in input_content]
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


def build_injected_user_message(command_text: str) -> dict:
    """Build the stream-json user message that injects a mid-run send_command
    turn into the running session via stdin."""
    return {
        "type": "user",
        "message": {"role": "user", "content": command_text},
        "parent_tool_use_id": None,
    }


def list_produced_artifacts(workspace_directory: str, baseline: "set | None") -> list:
    """List files under the workspace produced/changed since ``baseline``.

    Multimodal OUT: a claude run may write files into its workspace. We capture
    them by diffing the workspace file set against a baseline snapshot taken
    before the run. Returns absolute paths, sorted. The runner's own live-window
    files are excluded so they never look like produced artifacts.
    """
    from .live_files import (
        CONTROL_CHANNEL_FILE_NAME,
        LIVE_LOG_FILE_NAME,
        RUN_STATUS_FILE_NAME,
    )

    excluded = {LIVE_LOG_FILE_NAME, CONTROL_CHANNEL_FILE_NAME, RUN_STATUS_FILE_NAME}
    baseline = baseline or set()
    produced = []
    for root, _dirs, files in os.walk(workspace_directory):
        for name in files:
            full = os.path.join(root, name)
            if os.path.relpath(full, workspace_directory) in excluded:
                continue
            if name in excluded and root == workspace_directory:
                continue
            if full not in baseline:
                produced.append(full)
    return sorted(produced)


def snapshot_workspace_files(workspace_directory: str) -> set:
    """Snapshot the set of absolute file paths under the workspace (pre-run)."""
    snapshot = set()
    if not os.path.isdir(workspace_directory):
        return snapshot
    for root, _dirs, files in os.walk(workspace_directory):
        for name in files:
            snapshot.add(os.path.join(root, name))
    return snapshot

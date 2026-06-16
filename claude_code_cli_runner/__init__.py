"""claude-code-cli-runner: a generic, reusable runner for `claude -p` tasks.

Run a Claude Code CLI task as a STREAMING job, location-agnostic, with
multimodal input AND output. This package knows NOTHING about any particular
orchestrator: it has no task-return envelope, no composed-context, no
effort/thinking policy. It exposes one unified, always-streaming execution path
through :func:`run_claude_code_task`, plus a CLI and an optional streaming HTTP
server face — all driving the same core.

The on-disk live-window contract (file names + control intents) is fixed and
documented so external readers (e.g. a dashboard) can follow a run in real time.
"""

from __future__ import annotations

from .live_files import (
    CONTROL_CHANNEL_FILE_NAME,
    CONTROL_END_AND_RETURN,
    CONTROL_PAUSE,
    CONTROL_RESUME,
    CONTROL_SEND_COMMAND,
    LIVE_LOG_FILE_NAME,
    RUN_STATE_OPERATOR_ENDED,
    RUN_STATE_PAUSED,
    RUN_STATE_RUNNING,
    RUN_STATUS_FILE_NAME,
    control_channel_path,
    live_log_path,
    run_status_path,
)
from .request import (
    DocumentBlock,
    ImageBlock,
    ReusableContext,
    RunRequest,
    TextBlock,
)
from .result import RunResult
from .runner import run_claude_code_task

__all__ = [
    "run_claude_code_task",
    "RunRequest",
    "ReusableContext",
    "RunResult",
    "TextBlock",
    "ImageBlock",
    "DocumentBlock",
    "LIVE_LOG_FILE_NAME",
    "CONTROL_CHANNEL_FILE_NAME",
    "RUN_STATUS_FILE_NAME",
    "CONTROL_PAUSE",
    "CONTROL_RESUME",
    "CONTROL_SEND_COMMAND",
    "CONTROL_END_AND_RETURN",
    "RUN_STATE_RUNNING",
    "RUN_STATE_PAUSED",
    "RUN_STATE_OPERATOR_ENDED",
    "live_log_path",
    "control_channel_path",
    "run_status_path",
]

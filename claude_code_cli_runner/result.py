"""The run_result: the multimodal-aware outcome of a claude -p run.

Multimodal OUT: do not assume text-only. The result captures the assistant
text, any artifacts produced under the workspace, the parsed final result event,
exit/status, and the raw stream-log path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RunResult:
    """Outcome of one run_claude_code_task call.

    assistant_text: concatenated human-readable assistant text streamed back.
    final_result_event: the parsed ``{"type": "result", ...}`` chunk, if seen.
    produced_artifacts: absolute paths of files written under the workspace
      during the run (multimodal output: files, not just text).
    exit_code: the process exit code (None if terminated by us / unknown).
    operator_ended: True only when an end_and_return control intent ended it.
    live_log_path: absolute path of the raw stream-log JSONL file.
    run_state: the final out-of-band run-state annotation.
    workspace_directory: where the run happened / live files live.
    harness_stderr: the claude process's captured stderr. The actual error text
      when a run fails (e.g. an ``error_during_execution`` result carries no
      message); previously piped but never read, so failures surfaced only as a
      bare exit code. Empty on a clean run.
    """

    assistant_text: str = ""
    final_result_event: Optional[dict] = None
    produced_artifacts: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    operator_ended: bool = False
    live_log_path: str = ""
    run_state: str = ""
    workspace_directory: str = ""
    harness_stderr: str = ""

    def to_dict(self) -> dict:
        return {
            "assistant_text": self.assistant_text,
            "final_result_event": self.final_result_event,
            "produced_artifacts": self.produced_artifacts,
            "exit_code": self.exit_code,
            "operator_ended": self.operator_ended,
            "live_log_path": self.live_log_path,
            "run_state": self.run_state,
            "workspace_directory": self.workspace_directory,
            "harness_stderr": self.harness_stderr,
        }

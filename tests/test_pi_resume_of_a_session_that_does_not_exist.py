"""Pi can only resume a session that is in its session store. The pi integration says
whether one is, so the runner can start a fresh session instead of launching a resume
that exits with "No session found matching"."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner.pi_harness import PiHarnessIntegration
from claude_code_cli_runner.request import RunRequest, TextBlock


def _resume_request(workspace, session_id):
    return RunRequest(
        input_content=[TextBlock("continue")],
        workspace_directory=str(workspace),
        harness="pi",
        session_id=session_id,
        resume_session=True,
    )


def test_a_session_in_the_store_is_resumable(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    (tmp_path / "pi_sessions").mkdir()
    (tmp_path / "pi_sessions" / "2026-11-18T17-34-26-985Z_01a1efb9-64a9.jsonl").write_text("{}\n")
    assert PiHarnessIntegration().resumable_session_exists(
        _resume_request(tmp_path, "01a1efb9-64a9")
    ) is True


def test_a_session_from_another_harness_or_directory_is_not(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_SESSION_DIR", raising=False)
    (tmp_path / "pi_sessions").mkdir()
    (tmp_path / "pi_sessions" / "2026-11-18T17-34-26-985Z_01a1efb9-64a9.jsonl").write_text("{}\n")
    integration = PiHarnessIntegration()
    # a claude session id that pi never made
    assert integration.resumable_session_exists(
        _resume_request(tmp_path, "9495e0b5-c8cc-4a17-9176-919eaa0cf17b")
    ) is False
    # a directory with no session store at all
    assert integration.resumable_session_exists(
        _resume_request(tmp_path / "elsewhere", "01a1efb9-64a9")
    ) is False

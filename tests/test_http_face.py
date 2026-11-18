"""HTTP face round-trip (streaming) against the stub claude."""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import http_server
from conftest import stub_build_command


def test_http_streaming_round_trip(tmp_path, monkeypatch):
    # Point the runner (used inside the server thread) at the stub.
    monkeypatch.setattr(http_server, "run_claude_code_task", _runner_with_stub)

    os.environ["STUB_RESULT_TEXT"] = "http stub answer"
    server = http_server.build_streaming_http_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        outcome = http_server.post_run(
            "http://127.0.0.1:%d" % port,
            {
                "input_content": [{"block_type": "text", "text": "hello over http"}],
                "workspace_directory": str(tmp_path),
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        os.environ.pop("STUB_RESULT_TEXT", None)

    assert outcome["run_result"] is not None
    assert outcome["run_result"]["assistant_text"] == "http stub answer"
    # streamed live-log lines arrived before the final result line
    assert len(outcome["lines"]) >= 1


def test_resumed_run_streams_only_the_lines_this_run_appended(tmp_path, monkeypatch):
    """A run that RESUMES a session appends to the existing live log. The stream
    for that run must carry only the lines THIS run produced: re-sending the
    earlier runs' lines makes the caller treat old output as the current turn's
    (observed live 2026-11-18: a replayed 299-token reply tripped the caller's
    256-token per-turn limit, cancelling every follow-up turn before it began)."""
    monkeypatch.setattr(http_server, "run_claude_code_task", _runner_with_stub)

    prior_run_line = '{"received_at": 1.0, "chunk": {"type": "prior_run_marker"}}'
    (tmp_path / "task_live_log.jsonl").write_text(prior_run_line + "\n", encoding="utf-8")

    os.environ["STUB_RESULT_TEXT"] = "resumed stub answer"
    server = http_server.build_streaming_http_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        outcome = http_server.post_run(
            "http://127.0.0.1:%d" % port,
            {
                "input_content": [{"block_type": "text", "text": "follow-up question"}],
                "workspace_directory": str(tmp_path),
                "session_id": "11111111-1111-1111-1111-111111111111",
                "resume_session": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        os.environ.pop("STUB_RESULT_TEXT", None)

    assert outcome["run_result"] is not None
    assert len(outcome["lines"]) >= 1
    assert not any("prior_run_marker" in str(line) for line in outcome["lines"])
    # The on-disk log still holds the whole session history (resume appends).
    assert "prior_run_marker" in (tmp_path / "task_live_log.jsonl").read_text(encoding="utf-8")


def _runner_with_stub(run_request, **kwargs):
    from claude_code_cli_runner.runner import run_claude_code_task as real

    return real(run_request, build_command=stub_build_command)

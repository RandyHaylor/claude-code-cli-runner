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


def _runner_with_stub(run_request, **kwargs):
    from claude_code_cli_runner.runner import run_claude_code_task as real

    return real(run_request, build_command=stub_build_command)

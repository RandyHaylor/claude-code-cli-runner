"""HTTP /control endpoint + incremental stream_run client, against the stub."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import http_server
from conftest import stub_build_command


def _runner_with_stub(run_request, **kwargs):
    from claude_code_cli_runner.runner import run_claude_code_task as real

    return real(run_request, build_command=stub_build_command)


def _serve(monkeypatch):
    monkeypatch.setattr(http_server, "run_claude_code_task", _runner_with_stub)
    server = http_server.build_streaming_http_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread, "http://127.0.0.1:%d" % port


def test_stream_run_yields_incrementally_and_surfaces_run_id(tmp_path, monkeypatch):
    os.environ["STUB_RESULT_TEXT"] = "streamed answer"
    server, thread, base = _serve(monkeypatch)
    try:
        gen = http_server.stream_run(
            base,
            {
                "input_content": [{"block_type": "text", "text": "hi"}],
                "workspace_directory": str(tmp_path),
            },
        )
        first = next(gen)
        # First yielded item is the run-identity line.
        assert "run_started" in first
        assert first["run_started"]["workspace_token"] == str(tmp_path)

        live_lines = []
        run_result = None
        try:
            while True:
                live_lines.append(next(gen))
        except StopIteration as stop:
            run_result = stop.value
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        os.environ.pop("STUB_RESULT_TEXT", None)

    assert run_result is not None
    assert run_result["assistant_text"] == "streamed answer"
    assert len(live_lines) >= 1


def test_control_endpoint_writes_into_running_control_channel(tmp_path, monkeypatch):
    os.environ["STUB_RUN_FOREVER"] = "1"
    server, thread, base = _serve(monkeypatch)

    workspace = str(tmp_path)
    lines = []

    def consume():
        out = http_server.post_run(
            base,
            {
                "input_content": [{"block_type": "text", "text": "run forever"}],
                "workspace_directory": workspace,
            },
        )
        lines.append(out)

    runner = threading.Thread(target=consume)
    runner.start()
    try:
        # Wait until the run is live (control channel reachable), then end it.
        time.sleep(0.3)
        response = http_server.post_control(base, workspace, "end_and_return")
        assert response["control_written"]["control_intent"] == "end_and_return"
        runner.join(timeout=10)
    finally:
        os.environ.pop("STUB_RUN_FOREVER", None)
        server.shutdown()
        server.server_close()
        thread.join()

    assert lines, "run did not complete after end_and_return control"
    result = lines[0]["run_result"]
    assert result["operator_ended"] is True
    assert result["run_state"] == "operator_ended"


def test_control_send_command_validation(tmp_path, monkeypatch):
    import urllib.error

    server, thread, base = _serve(monkeypatch)
    try:
        # send_command with no command_text is rejected (400).
        raised = False
        try:
            http_server.post_control(base, str(tmp_path), "send_command")
        except urllib.error.HTTPError as err:
            raised = err.code == 400
        assert raised
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

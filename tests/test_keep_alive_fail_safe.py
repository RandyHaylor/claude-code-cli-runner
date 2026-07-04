"""Keep-alive fail-safe (nd-372): the runner's duty to kill an ORPHANED harness.

Three layers are proven here:
  * the in-memory keep-alive registry (record / age / clear);
  * the HTTP POST /keep-alive/<task_id> endpoint records a signal's arrival;
  * a run launched with keep_alive_expected is aggressively killed when the
    relayed signal stops for the timeout, and is NOT killed while signals arrive.
"""

import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import RunRequest, TextBlock, http_server, keep_alive_registry
from claude_code_cli_runner.runner import run_claude_code_task
from conftest import stub_build_command


# -- the registry ------------------------------------------------------------
def test_registry_records_ages_and_clears():
    keep_alive_registry.clear_keep_alive_record("reg_task")
    assert keep_alive_registry.seconds_since_last_keep_alive("reg_task") is None
    keep_alive_registry.record_keep_alive_signal("reg_task")
    age = keep_alive_registry.seconds_since_last_keep_alive("reg_task")
    assert age is not None and age >= 0.0
    keep_alive_registry.clear_keep_alive_record("reg_task")
    assert keep_alive_registry.seconds_since_last_keep_alive("reg_task") is None


# -- the HTTP endpoint -------------------------------------------------------
def _serve():
    server = http_server.build_streaming_http_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread, "http://127.0.0.1:%d" % port


def test_keep_alive_endpoint_records_a_signal_for_the_task_id():
    task_id = "endpoint_task_xyz"
    keep_alive_registry.clear_keep_alive_record(task_id)
    server, thread, base = _serve()
    try:
        request = urllib.request.Request(
            base + "/keep-alive/" + task_id, data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["keep_alive_recorded"] == task_id
        # The serve recorded the arrival in its process-local registry.
        assert keep_alive_registry.seconds_since_last_keep_alive(task_id) is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        keep_alive_registry.clear_keep_alive_record(task_id)


# -- the watchdog kill -------------------------------------------------------
_STREAM_THEN_SLEEP_STUB = """\
import sys, time
sys.stdin.readline()  # consume the prompt like a real harness
print('{"type": "assistant", "message": {"content": [{"type": "text", '
      '"text": "starting"}]}}', flush=True)
time.sleep(120)  # keep running far past the keep-alive timeout
"""


def _stream_then_sleep_build_command(tmp_path):
    stub_path = os.path.join(str(tmp_path), "stub_stream_then_sleep.py")
    with open(stub_path, "w", encoding="utf-8") as handle:
        handle.write(_STREAM_THEN_SLEEP_STUB)

    def build_command(run_request):
        return [sys.executable, stub_path]

    return build_command


def test_orphaned_run_is_killed_when_keep_alive_stops(tmp_path):
    task_id = "orphan_task_1"
    keep_alive_registry.clear_keep_alive_record(task_id)
    request = RunRequest(
        input_content=[TextBlock(text="do something")],
        workspace_directory=str(tmp_path),
        keep_alive_expected=True,
        keep_alive_timeout_seconds=1.5,
        keep_alive_task_id=task_id,
        # idle_kill deliberately unset: only the keep-alive watchdog is on trial.
    )
    started = time.monotonic()
    result = run_claude_code_task(
        request, build_command=_stream_then_sleep_build_command(tmp_path)
    )
    elapsed = time.monotonic() - started

    assert elapsed < 30, "the keep-alive watchdog must kill long before the stub sleep"
    assert result.run_state == "keep_alive_lost_killed"
    assert "keep-alive" in (result.harness_stderr or "")
    with open(result.live_log_path, encoding="utf-8") as handle:
        log_lines = [json.loads(line) for line in handle if line.strip()]
    assert any("runner_keep_alive_lost_kill" in record for record in log_lines)


def test_run_is_not_killed_while_keep_alive_signals_keep_arriving(tmp_path):
    task_id = "healthy_task_1"
    keep_alive_registry.clear_keep_alive_record(task_id)
    stop_feeding = threading.Event()

    def feed_keep_alive_signals():
        # Relayed beats keep arriving well within the timeout, so the watchdog
        # must never fire — this is the "healthy run is NOT killed" guarantee.
        while not stop_feeding.wait(0.4):
            keep_alive_registry.record_keep_alive_signal(task_id)

    feeder = threading.Thread(target=feed_keep_alive_signals)
    feeder.start()

    # A stub that runs briefly (silent) then finishes normally.
    running_stub = os.path.join(str(tmp_path), "stub_brief.py")
    with open(running_stub, "w", encoding="utf-8") as handle:
        handle.write(
            "import sys, time\n"
            "sys.stdin.readline()\n"
            "time.sleep(3.0)\n"
            "print('{\"type\": \"assistant\", \"message\": {\"content\": "
            "[{\"type\": \"text\", \"text\": \"finished normally\"}]}}', flush=True)\n"
        )

    def build_command(run_request):
        return [sys.executable, running_stub]

    request = RunRequest(
        input_content=[TextBlock(text="hello")],
        workspace_directory=str(tmp_path),
        keep_alive_expected=True,
        keep_alive_timeout_seconds=1.5,
        keep_alive_task_id=task_id,
    )
    try:
        result = run_claude_code_task(request, build_command=build_command)
    finally:
        stop_feeding.set()
        feeder.join()
        keep_alive_registry.clear_keep_alive_record(task_id)

    assert result.run_state != "keep_alive_lost_killed"
    assert result.assistant_text == "finished normally"

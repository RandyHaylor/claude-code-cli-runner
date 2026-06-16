"""Optional HTTP server face: STREAMING-capable, driving the same core.

Unlike a blocking passthrough server, this runs the task through
run_claude_code_task in a background thread and STREAMS the live-log JSONL back
to the client as it grows (chunked text/plain), so an HTTP client can follow the
run in real time. When the run finishes the final line is a JSON RunResult.

Wire format:
    POST /run
      body JSON: a run_request (see ``request_from_json``)
      200 streaming response: a FIRST line ``{"run_started": {"run_id": ...,
          "workspace_token": ...}}`` identifying the run, then live-log JSONL
          lines as they arrive, then a final line ``{"run_result": {...}}``.
    POST /control
      body JSON: ``{"run_id"|"workspace_token": ..., "control_intent": ...,
          "command_text"?: ...}`` — writes the intent into THAT run's control
          channel (pause/resume/send_command/end_and_return). 200
          ``{"control_written": {...}}`` on success.
    401 when an access token is configured and the X-Access-Token header is bad.
    400 malformed body; 404 any other method/path.

RUN IDENTITY (simplest correct scheme): a run is identified by its
``workspace_token``, which IS its ``workspace_directory`` (the absolute path the
control channel lives in). ``run_id`` is accepted as an alias. ``/control`` simply
appends the intent into that workspace's control channel; the running task drains
it out of band. No server-side run registry is needed because the control channel
is already a per-workspace on-disk rendezvous.

Clients:
  :func:`stream_run`  - generator yielding each streamed line incrementally
                        (live-log records as they arrive), returning the parsed
                        run_result via StopIteration.value.
  :func:`post_run`    - convenience wrapper collecting all lines + run_result.
  :func:`post_control`- POST a control intent for an active run.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .request import (
    DocumentBlock,
    ImageBlock,
    ReusableContext,
    RunRequest,
    SshConfig,
    TextBlock,
)
from .live_files import append_control_intent
from .runner import run_claude_code_task

_BLOCK_BUILDERS = {
    "text": lambda d: TextBlock(text=d.get("text", "")),
    "image": lambda d: ImageBlock(
        mime_type=d["mime_type"], path=d.get("path"), data_base64=d.get("data_base64")
    ),
    "document": lambda d: DocumentBlock(
        mime_type=d["mime_type"],
        path=d.get("path"),
        data_base64=d.get("data_base64"),
        name=d.get("name"),
    ),
}


def _blocks_from_json(blocks) -> list:
    built = []
    for block in blocks or []:
        builder = _BLOCK_BUILDERS.get(block.get("block_type") or block.get("type"))
        if builder is None:
            raise ValueError("unknown content block: %r" % block)
        built.append(builder(block))
    return built


def request_from_json(payload: dict) -> RunRequest:
    """Build a RunRequest from the JSON wire shape."""
    input_content = _blocks_from_json(payload.get("input_content", []))

    ssh = None
    if payload.get("ssh"):
        ssh = SshConfig(**payload["ssh"])

    reusable_context = None
    rc = payload.get("reusable_context")
    if rc:
        reusable_context = ReusableContext(
            chunk_id=rc["chunk_id"],
            content=_blocks_from_json(rc.get("content", [])),
        )

    return RunRequest(
        input_content=input_content,
        workspace_directory=payload["workspace_directory"],
        model=payload.get("model"),
        execution_location=payload.get("execution_location", "local_subprocess"),
        ssh=ssh,
        dangerously_skip_permissions=payload.get("dangerously_skip_permissions", False),
        extra_cli_flags=payload.get("extra_cli_flags", []),
        claude_command=payload.get("claude_command", "claude"),
        timeout_seconds=payload.get("timeout_seconds"),
        reusable_context=reusable_context,
        enable_session_reuse=payload.get("enable_session_reuse", True),
    )


def build_streaming_http_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    access_token: "str | None" = None,
    poll_seconds: float = 0.02,
) -> ThreadingHTTPServer:
    """Return a configured (not yet serving) streaming HTTP server."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def _error(self, status, message):
            body = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._error(404, "not found")

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _token_ok(self):
            if access_token is None:
                return True
            return self.headers.get("X-Access-Token") == access_token

        def do_POST(self):
            if not self._token_ok():
                self._error(401, "missing or invalid access token")
                return
            if self.path == "/control":
                self._handle_control()
                return
            if self.path != "/run":
                self._error(404, "not found")
                return
            try:
                payload = self._read_json_body()
                run_request = request_from_json(payload)
            except (ValueError, KeyError, UnicodeDecodeError) as cause:
                self._error(400, "bad request: %s" % cause)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()

            # FIRST line surfaces the run identity the caller uses for /control.
            workspace_token = run_request.workspace_directory
            started = {
                "run_started": {
                    "run_id": workspace_token,
                    "workspace_token": workspace_token,
                }
            }
            try:
                self.wfile.write((json.dumps(started) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, OSError):
                return

            result_holder = {}

            def run_it():
                result_holder["result"] = run_claude_code_task(run_request)

            worker = threading.Thread(target=run_it)
            worker.start()

            # Stream the live log as it grows, then the final run_result line.
            log_path = os.path.join(
                run_request.workspace_directory, "task_live_log.jsonl"
            )
            offset = 0
            while worker.is_alive() or offset is not None:
                offset = self._flush_new_lines(log_path, offset)
                if not worker.is_alive():
                    # Drain whatever remains, then finish.
                    self._flush_new_lines(log_path, offset)
                    break
                time.sleep(poll_seconds)

            worker.join()
            result = result_holder.get("result")
            final = {"run_result": result.to_dict() if result else None}
            try:
                self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, OSError):
                pass

        def _flush_new_lines(self, log_path, offset):
            if not os.path.isfile(log_path):
                return offset
            with open(log_path, "r", encoding="utf-8") as handle:
                handle.seek(offset)
                data = handle.read()
                new_offset = handle.tell()
            if data:
                try:
                    self.wfile.write(data.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, OSError):
                    pass
            return new_offset

        def _handle_control(self):
            try:
                payload = self._read_json_body()
            except (ValueError, UnicodeDecodeError) as cause:
                self._error(400, "bad request: %s" % cause)
                return
            workspace_token = payload.get("workspace_token") or payload.get("run_id")
            control_intent = payload.get("control_intent")
            command_text = payload.get("command_text")
            if not workspace_token or not control_intent:
                self._error(
                    400, "control requires 'workspace_token' (or 'run_id') and 'control_intent'"
                )
                return
            try:
                written = append_control_intent(
                    workspace_token, control_intent, command_text=command_text
                )
            except ValueError as cause:
                self._error(400, "bad control intent: %s" % cause)
                return
            body = json.dumps({"control_written": written}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)


def run_streaming_http_server(
    host: str = "127.0.0.1", port: int = 8765, access_token: "str | None" = None
) -> None:
    """Blocking: build the streaming server and serve forever."""
    server = build_streaming_http_server(host=host, port=port, access_token=access_token)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def stream_run(server_url: str, run_request_json: dict, access_token: "str | None" = None):
    """STREAMING client: POST a run_request and YIELD each streamed line as it
    arrives (incrementally, NOT buffered until completion), so a caller can tee
    the live log in real time.

    Yields, in order:
      - ``{"run_started": {"run_id", "workspace_token"}}`` (first line),
      - each live-log JSONL record (a dict) as it is produced,
      - any unparseable raw line as a str.
    Does NOT yield the final ``{"run_result": ...}`` line; instead it is the
    generator's return value (available via ``StopIteration.value`` or a
    ``yield from`` target). :func:`post_run` shows the collect-all pattern.
    """
    body = json.dumps(run_request_json).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if access_token:
        request.add_header("X-Access-Token", access_token)
    run_result = None
    with urllib.request.urlopen(request) as response:
        for raw in response:
            text = raw.decode("utf-8").rstrip("\n")
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                yield text
                continue
            if isinstance(parsed, dict) and "run_result" in parsed:
                run_result = parsed["run_result"]
                continue
            yield parsed
    return run_result


def post_run(server_url: str, run_request_json: dict, access_token: "str | None" = None) -> dict:
    """Convenience client: drive :func:`stream_run` to completion, collecting the
    streamed live-log lines, and return ``{"lines": [...], "run_result": {...},
    "run_id": ...}``. The first ``run_started`` line is split out into ``run_id``."""
    lines = []
    run_id = None
    generator = stream_run(server_url, run_request_json, access_token=access_token)
    try:
        while True:
            item = next(generator)
            if isinstance(item, dict) and "run_started" in item:
                run_id = item["run_started"].get("run_id")
                continue
            lines.append(item)
    except StopIteration as stop:
        run_result = stop.value
    return {"lines": lines, "run_result": run_result, "run_id": run_id}


def post_control(
    server_url: str,
    workspace_token: str,
    control_intent: str,
    command_text: "str | None" = None,
    access_token: "str | None" = None,
) -> dict:
    """Client: drive a control intent for an active run via ``POST /control``.

    ``workspace_token`` is the run identity surfaced by the ``run_started`` line
    (== the run's workspace_directory). Returns the server's JSON response."""
    payload = {"workspace_token": workspace_token, "control_intent": control_intent}
    if command_text is not None:
        payload["command_text"] = command_text
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/control",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if access_token:
        request.add_header("X-Access-Token", access_token)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))

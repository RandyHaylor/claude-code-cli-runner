"""Optional HTTP server face: STREAMING-capable, driving the same core.

Unlike a blocking passthrough server, this runs the task through
run_claude_code_task in a background thread and STREAMS the live-log JSONL back
to the client as it grows (chunked text/plain), so an HTTP client can follow the
run in real time. When the run finishes the final line is a JSON RunResult.

Wire format:
    POST /run
      body JSON: a run_request (see ``request_from_json``)
      200 streaming response: live-log JSONL lines as they arrive, then a final
          line ``{"run_result": {...}}``.
    401 when an access token is configured and the X-Access-Token header is bad.
    400 malformed body; 404 any other method/path.

A small client, :func:`post_run`, posts a request and returns the streamed lines
plus the parsed run_result.
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
    RunRequest,
    SshConfig,
    TextBlock,
)
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


def request_from_json(payload: dict) -> RunRequest:
    """Build a RunRequest from the JSON wire shape."""
    input_content = []
    for block in payload.get("input_content", []):
        builder = _BLOCK_BUILDERS.get(block.get("block_type") or block.get("type"))
        if builder is None:
            raise ValueError("unknown content block: %r" % block)
        input_content.append(builder(block))

    ssh = None
    if payload.get("ssh"):
        ssh = SshConfig(**payload["ssh"])

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

        def do_POST(self):
            if self.path != "/run":
                self._error(404, "not found")
                return
            if access_token is not None and self.headers.get("X-Access-Token") != access_token:
                self._error(401, "missing or invalid access token")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                run_request = request_from_json(payload)
            except (ValueError, KeyError, UnicodeDecodeError) as cause:
                self._error(400, "bad request: %s" % cause)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()

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


def post_run(server_url: str, run_request_json: dict, access_token: "str | None" = None) -> dict:
    """Small client: POST a run_request to the streaming server, collect the
    streamed live-log lines, and return ``{"lines": [...], "run_result": {...}}``."""
    body = json.dumps(run_request_json).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if access_token:
        request.add_header("X-Access-Token", access_token)
    lines = []
    run_result = None
    with urllib.request.urlopen(request) as response:
        for raw in response:
            text = raw.decode("utf-8").rstrip("\n")
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                lines.append(text)
                continue
            if isinstance(parsed, dict) and "run_result" in parsed:
                run_result = parsed["run_result"]
            else:
                lines.append(parsed)
    return {"lines": lines, "run_result": run_result}

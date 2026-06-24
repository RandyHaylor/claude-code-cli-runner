"""raw-515: the harness auth-status check — check_claude_auth_status() normalizes
`claude auth status --json`, and the GET /auth-status serve verb returns it. No
real `claude` CLI is invoked: the subprocess is stubbed."""

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner import check_claude_auth_status, http_server
from claude_code_cli_runner.auth_status import CLAUDE_AUTH_STATUS_COMMAND


class _CompletedProcessStub:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_returning(stdout, returncode=0, stderr=""):
    def run_subprocess(argv, **kwargs):
        run_subprocess.seen_argv = argv
        return _CompletedProcessStub(returncode, stdout, stderr)
    return run_subprocess


LOGGED_IN_JSON = json.dumps({
    "loggedIn": True, "authMethod": "claude.ai", "email": "a@b.com",
    "orgId": "o1", "orgName": "Acme", "subscriptionType": "team",
})


def test_logged_in_is_normalized():
    stub = _stub_returning(LOGGED_IN_JSON, returncode=0)
    status = check_claude_auth_status(run_subprocess=stub)
    assert status["logged_in"] is True
    assert status["email"] == "a@b.com"
    assert status["org_name"] == "Acme"
    assert status["subscription_type"] == "team"
    assert status["error"] is None
    # It used the real, dedicated CLI auth-check command.
    assert stub.seen_argv == CLAUDE_AUTH_STATUS_COMMAND


def test_logged_out_is_normalized():
    stub = _stub_returning(json.dumps({"loggedIn": False}), returncode=0)
    status = check_claude_auth_status(run_subprocess=stub)
    assert status["logged_in"] is False
    assert status["error"] is None


def test_unparseable_output_is_not_logged_in_with_error():
    stub = _stub_returning("not json at all", returncode=1)
    status = check_claude_auth_status(run_subprocess=stub)
    assert status["logged_in"] is False
    assert status["error"]


def test_missing_cli_is_reported_not_raised():
    def run_subprocess(argv, **kwargs):
        raise FileNotFoundError("claude")
    status = check_claude_auth_status(run_subprocess=run_subprocess)
    assert status["logged_in"] is False
    assert "not found" in status["error"]


def test_timeout_is_reported_not_raised():
    def run_subprocess(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=30)
    status = check_claude_auth_status(run_subprocess=run_subprocess)
    assert status["logged_in"] is False
    assert "timed out" in status["error"]


def _serve(monkeypatch, canned_status, access_token=None):
    monkeypatch.setattr(
        http_server, "check_claude_auth_status", lambda: canned_status
    )
    server = http_server.build_streaming_http_server(
        host="127.0.0.1", port=0, access_token=access_token
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread, "http://127.0.0.1:%d" % port


def test_http_auth_status_returns_canned_status(monkeypatch):
    canned = {"logged_in": True, "email": "a@b.com", "error": None}
    server, thread, base = _serve(monkeypatch, canned)
    try:
        status = http_server.fetch_auth_status(base)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert status == canned


def test_http_auth_status_is_token_gated(monkeypatch):
    canned = {"logged_in": True, "error": None}
    server, thread, base = _serve(monkeypatch, canned, access_token="s3cret")
    try:
        # No token -> 401.
        try:
            http_server.fetch_auth_status(base)
            raised = None
        except urllib.error.HTTPError as failure:
            raised = failure.code
        # With token -> 200.
        ok = http_server.fetch_auth_status(base, access_token="s3cret")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert raised == 401
    assert ok == canned

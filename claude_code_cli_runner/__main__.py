"""CLI face: `python3 -m claude_code_cli_runner ...` (also the console script).

Drives the SAME core (run_claude_code_task) as the library and HTTP faces.

Subcommands:
  run       run a task and print a JSON RunResult
  serve     start the streaming HTTP server face
  control   append a control intent to a task's control channel
"""

from __future__ import annotations

import argparse
import json
import sys

from .live_files import KNOWN_CONTROL_INTENTS, append_control_intent
from .request import (
    DocumentBlock,
    ImageBlock,
    RunRequest,
    SshConfig,
    TextBlock,
)
from .runner import run_claude_code_task


def _build_request_from_args(args) -> RunRequest:
    input_content = []
    for text in args.text or []:
        input_content.append(TextBlock(text=text))
    for spec in args.image or []:
        mime, _, path = spec.partition(":")
        input_content.append(ImageBlock(mime_type=mime, path=path))
    for spec in args.document or []:
        mime, _, path = spec.partition(":")
        input_content.append(DocumentBlock(mime_type=mime, path=path))

    ssh = None
    if args.execution_location in ("vm_over_ssh", "remote_host"):
        ssh = SshConfig(
            host=args.ssh_host,
            user=args.ssh_user,
            key_path=args.ssh_key,
            port=args.ssh_port,
            vm_name=args.vm_name,
            remote_workspace_directory=args.remote_workspace,
        )
    return RunRequest(
        input_content=input_content,
        workspace_directory=args.workspace,
        model=args.model,
        execution_location=args.execution_location,
        ssh=ssh,
        dangerously_skip_permissions=args.dangerously_skip_permissions,
        extra_cli_flags=args.cli_flag or [],
        claude_command=args.claude_command,
        timeout_seconds=args.timeout_seconds,
    )


def _add_run_args(parser):
    parser.add_argument("--text", action="append", help="an inline text block (repeatable)")
    parser.add_argument(
        "--image", action="append", help="an image block as MIME:PATH (repeatable)"
    )
    parser.add_argument(
        "--document", action="append", help="a document block as MIME:PATH (repeatable)"
    )
    parser.add_argument("--workspace", required=True, help="workspace directory")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--execution-location",
        dest="execution_location",
        default="local_subprocess",
        choices=["local_subprocess", "vm_over_ssh", "remote_host"],
    )
    parser.add_argument("--ssh-host", dest="ssh_host", default=None)
    parser.add_argument("--ssh-user", dest="ssh_user", default="agent-user")
    parser.add_argument("--ssh-key", dest="ssh_key", default=None)
    parser.add_argument("--ssh-port", dest="ssh_port", type=int, default=22)
    parser.add_argument("--vm-name", dest="vm_name", default=None)
    parser.add_argument("--remote-workspace", dest="remote_workspace", default=None)
    parser.add_argument(
        "--dangerously-skip-permissions",
        dest="dangerously_skip_permissions",
        action="store_true",
    )
    parser.add_argument("--cli-flag", action="append", help="extra raw claude flag (repeatable)")
    parser.add_argument("--claude-command", dest="claude_command", default="claude")
    parser.add_argument("--timeout-seconds", dest="timeout_seconds", type=float, default=None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="claude_code_cli_runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a claude -p task")
    _add_run_args(run_parser)

    serve_parser = sub.add_parser("serve", help="start the streaming HTTP server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--access-token", dest="access_token", default=None)

    control_parser = sub.add_parser("control", help="append a control intent")
    control_parser.add_argument("--workspace", required=True)
    control_parser.add_argument("--intent", required=True, choices=list(KNOWN_CONTROL_INTENTS))
    control_parser.add_argument("--command-text", dest="command_text", default=None)

    args = parser.parse_args(argv)

    if args.command == "run":
        run_request = _build_request_from_args(args)
        result = run_claude_code_task(run_request)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "serve":
        from .http_server import run_streaming_http_server

        run_streaming_http_server(
            host=args.host, port=args.port, access_token=args.access_token
        )
        return 0

    if args.command == "control":
        intent = append_control_intent(args.workspace, args.intent, args.command_text)
        print(json.dumps(intent))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())

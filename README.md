# claude-code-cli-runner

A fully generic, reusable **streaming runner for `claude -p` tasks**:
location-agnostic, with **multimodal input AND output**. One unified,
always-streaming execution path, exposed through three faces — a library API, a
CLI, and an optional streaming HTTP server — all driving the same core.

This tool knows nothing about any particular orchestrator. There is no
task-return envelope, no composed-context, and no effort/thinking policy here —
those are deliberately left to a thin glue adapter that depends on this package
(see "Deliberately out of scope" below).

The only external runtime dependency is the `claude` CLI itself. Python 3.10+,
standard library only. Tests run entirely against a stub `claude` — never the
real CLI.

## Install

```bash
pip install -e .
```

## The three faces

All three drive the same core function, `run_claude_code_task`.

### 1. Library / import API

```python
from claude_code_cli_runner import (
    RunRequest, TextBlock, ImageBlock, DocumentBlock, run_claude_code_task,
)

result = run_claude_code_task(RunRequest(
    input_content=[
        TextBlock(text="Describe this image."),
        ImageBlock(mime_type="image/png", path="/path/to/pic.png"),
    ],
    workspace_directory="/path/to/workspace",
    model="some-model",                 # optional
    execution_location="local_subprocess",
    dangerously_skip_permissions=True,  # explicit opt-in
))

print(result.assistant_text)
print(result.produced_artifacts)
```

Entry point: `claude_code_cli_runner.run_claude_code_task(run_request, *, build_command=None)`.

### 2. CLI

```bash
python3 -m claude_code_cli_runner run \
    --text "Say hello." \
    --image image/png:/path/to/pic.png \
    --document application/pdf:/path/to/spec.pdf \
    --workspace /path/to/workspace \
    --execution-location local_subprocess \
    --dangerously-skip-permissions

python3 -m claude_code_cli_runner serve --host 127.0.0.1 --port 8765
python3 -m claude_code_cli_runner control --workspace /ws --intent pause
```

Entry point: `claude_code_cli_runner.__main__:main` (also the
`claude-code-cli-runner` console script).

### 3. Optional streaming HTTP server

```bash
python3 -m claude_code_cli_runner serve --port 8765
```

```python
from claude_code_cli_runner.http_server import post_run
outcome = post_run("http://127.0.0.1:8765", {
    "input_content": [{"block_type": "text", "text": "hello"}],
    "workspace_directory": "/path/to/workspace",
})
# outcome["lines"]      -> the live-log records streamed as the run progressed
# outcome["run_result"] -> the final RunResult dict
```

`POST /run` streams the live-log JSONL lines back as they arrive (chunked), then
a final `{"run_result": {...}}` line. Entry points:
`build_streaming_http_server`, `run_streaming_http_server`, `post_run`.

## run_request shape

`RunRequest` (`claude_code_cli_runner.request`):

| field | meaning |
|---|---|
| `input_content` | LIST of content blocks (multimodal, see below) |
| `workspace_directory` | cwd for a local run; where the live files always land (on the orchestrating host) |
| `model` | optional model id (`--model`) |
| `execution_location` | `local_subprocess` \| `vm_over_ssh` \| `remote_host` |
| `ssh` | `SshConfig` for the remote transports |
| `dangerously_skip_permissions` | explicit opt-in to `--dangerously-skip-permissions` |
| `extra_cli_flags` | extra raw argv flags appended to the claude command |
| `claude_command` | executable name/path (default `"claude"`; tests use a stub) |
| `live_log_path` / `control_channel_path` / `run_status_path` | optional explicit paths; default to the contract names under the workspace |
| `timeout_seconds` | optional overall wall-clock budget |

### Multimodal input — content block shapes

Adding a modality is **data, not code**: each block is a small dataclass with a
`block_type`, serialized into the `claude -p` stream-json user message `content`
array.

```python
TextBlock(text="...")
ImageBlock(mime_type="image/png", path="...")          # or data_base64="..."
DocumentBlock(mime_type="application/pdf", path="...",  # or data_base64="..."
              name="spec.pdf")
```

Serialized entries:

```jsonc
{"type": "text", "text": "..."}
{"type": "image",    "source": {"type": "base64", "media_type": "image/png",       "data": "<b64>"}}
{"type": "document", "source": {"type": "base64", "media_type": "application/pdf",  "data": "<b64>"}, "title": "spec.pdf"}
```

## run_result shape

`RunResult` (`claude_code_cli_runner.result`) — multimodal-aware, never assumes
text-only output:

| field | meaning |
|---|---|
| `assistant_text` | concatenated assistant text streamed back |
| `final_result_event` | the parsed `{"type": "result", ...}` chunk, if seen |
| `produced_artifacts` | absolute paths of files the run wrote under the workspace |
| `exit_code` | process exit code (None if we terminated it) |
| `operator_ended` | True only when an `end_and_return` intent ended the run |
| `live_log_path` | absolute path of the raw stream-log JSONL |
| `run_state` | final out-of-band run-state annotation |
| `workspace_directory` | where the run happened / live files live |

## execution_location config — location is config, not code

```python
from claude_code_cli_runner.request import SshConfig

# local
execution_location="local_subprocess"

# over SSH (VM or any remote host)
execution_location="vm_over_ssh"   # or "remote_host"
ssh=SshConfig(
    host="10.0.0.5",        # OR leave None and set vm_name to resolve via libvirt DHCP
    user="agent-user",
    key_path="/path/to/key",
    port=22,
    vm_name=None,           # optional libvirt-DHCP lookup, fully overridable
    remote_workspace_directory=None,  # cd here on the remote host before running claude
)
```

Every location resolves to a `build_command(run_request) -> argv` and feeds the
**same** streaming runner. The local transport runs `claude` directly; the SSH
transports wrap the identical claude argv in an `ssh` invocation. The prompt is
never on argv — it is delivered as a stdin stream-json user message — so it never
lands on a process table, locally or remotely.

## Live-window file contract

Under the workspace directory the runner maintains three files an external
reader (e.g. a dashboard) can follow in real time. **These exact names and
control intents are the public contract:**

- `task_live_log.jsonl` — append-only JSONL; one record per stream chunk
- `task_control_channel.jsonl` — append-only JSONL an external reader **writes**
  control intents into
- `task_run_status.json` — sidecar reflecting the out-of-band run state
  (`running` / `paused` / `operator_ended`)

Control intents (write a JSONL line into the control channel):

- `{"control_intent": "pause"}`
- `{"control_intent": "resume"}`
- `{"control_intent": "send_command", "command_text": "..."}`
- `{"control_intent": "end_and_return"}`

## Examples (offline, against the stub)

```bash
python3 examples/run_local_text.py
python3 examples/run_multimodal.py
```

## Tests

```bash
python3 -m pytest tests/ -q
```

Every test runs against `tests/stub_streaming_claude.py` — the real `claude` CLI
is never invoked.

## Deliberately out of scope (orchestrator-glue concerns)

This package is intentionally generic. The following live in a thin glue adapter
that depends on this tool, NOT here:

- any task-return **envelope** shape / parsing
- **composed-context** prompt assembly
- **effort/thinking** (creativity) -> claude settings translation / policy
- any concept of containers, definition-of-done, or queue/runner states

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

For the design rationale and module-level walkthrough, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- Python 3.10+ (standard library only — no third-party runtime dependencies).
- A working `claude` CLI on `PATH` (or pointed at via `claude_command`) for
  **real** runs. The test suite needs none of this — it runs against a stub.

## Install

```bash
# from a local checkout
pip install -e .

# or straight from git
pip install "git+https://github.com/RandyHaylor/claude-code-cli-runner.git"
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
from claude_code_cli_runner.http_server import post_run, stream_run, post_control

# Buffered convenience client:
outcome = post_run("http://127.0.0.1:8765", {
    "input_content": [{"block_type": "text", "text": "hello"}],
    "workspace_directory": "/path/to/workspace",
})
# outcome["run_id"]     -> the run identity (== workspace_token) for /control
# outcome["lines"]      -> the live-log records streamed as the run progressed
# outcome["run_result"] -> the final RunResult dict

# INCREMENTAL streaming client: yields each line as it arrives (real-time tee):
gen = stream_run("http://127.0.0.1:8765", request_json)
for item in gen:           # run_started line, then each live-log record
    handle(item)
# the final run_result is the generator's return value (StopIteration.value)
```

`POST /run` first emits a `{"run_started": {"run_id", "workspace_token"}}` line
identifying the run, then streams the live-log JSONL lines as they arrive
(chunked), then a final `{"run_result": {...}}` line.

**Driving an in-flight run over HTTP** — `POST /control`:

```python
post_control("http://127.0.0.1:8765", workspace_token, "pause")
post_control("http://127.0.0.1:8765", workspace_token, "send_command",
             command_text="also check the logs")
post_control("http://127.0.0.1:8765", workspace_token, "end_and_return")
```

**Run identity (simplest correct scheme):** a run is identified by its
`workspace_token`, which IS its `workspace_directory` (where the control channel
lives). `run_id` is accepted as an alias. `/control` simply appends the intent
into that workspace's `task_control_channel.jsonl`; the running task drains it out
of band. No server-side registry is needed because the per-workspace control
channel is already the rendezvous. (Limitation: a remote caller must therefore
know/agree on the workspace path; if two runs ever shared a workspace, controls
would target whichever run is reading that channel — runs should use distinct
workspaces.)

Entry points: `build_streaming_http_server`, `run_streaming_http_server`,
`stream_run` (incremental), `post_run` (buffered), `post_control`.

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
| `reusable_context` | optional leading `ReusableContext(chunk_id, content=[blocks])` — see below |
| `enable_session_reuse` | best-effort opt-out (default `True`); `False` always inlines the chunk |

### Optional reusable context (prime-once / fork-per-task) — best-effort

A caller MAY mark a **leading** context chunk as repeatable by attaching a
`ReusableContext` with a stable, unique `chunk_id` and its own list of content
blocks (same block types as `input_content`):

```python
from claude_code_cli_runner import ReusableContext, RunRequest, TextBlock

RunRequest(
    input_content=[TextBlock(text="...the per-task remainder...")],
    workspace_directory="/work/run-1",
    reusable_context=ReusableContext(
        chunk_id="house-style-v3",
        content=[TextBlock(text="...a big leading context shared by many tasks...")],
    ),
    enable_session_reuse=True,   # default
)
```

Wire shape (`POST /run` JSON), same block types as `input_content`:

```jsonc
{
  "input_content": [ {"block_type": "text", "text": "the per-task remainder"} ],
  "workspace_directory": "/work/run-1",
  "reusable_context": {
    "chunk_id": "house-style-v3",
    "content": [ {"block_type": "text", "text": "big leading context"} ]
  },
  "enable_session_reuse": true
}
```

**Prime-once / fork model.** When a `reusable_context` is present AND
`enable_session_reuse` is `True`, the runner:

1. **Primes once** per `chunk_id`: if the id is not yet registered, it runs a
   throwaway, **self-completing** `claude --session-id <new_primed_sid> -p
   "<chunk text>"` priming session — a plain `-p` call with the chunk as the
   positional prompt and **no** stream-json flags and **no** stdin, so claude
   ingests the chunk and exits 0 on its own. On exit 0 it records
   `chunk_id -> primed_sid` in the on-disk registry. (The earlier streaming
   prime over stdin never produced a result and so never persisted a session;
   the simple completing call does.) Only **text-only** chunks are primed this
   way; a chunk containing image/document blocks cannot be carried as a `-p`
   positional prompt, so it is **not** primed and the run falls back to inline
   (multimodal chunk priming is a deferred enhancement).
2. **Forks per task** (cross-cwd capable): it runs the task as
   `claude --resume <primed_sid> --fork-session --session-id <fresh_task_sid> -p`,
   sending ONLY the per-task `input_content` over stdin. The chunk is already in
   the primed session, so its tokens are cache-reused and **not re-sent**. The
   primed session is untouched and reusable; later tasks with the same
   `chunk_id` skip priming and fork directly.

**Cross-cwd reuse via session-jsonl relocation.** claude stores each session at
`~/.claude/projects/<cwd-encoded>/<session-id>.jsonl`, where `<cwd-encoded>` is
the absolute cwd with every `/` **and** `_` replaced by `-` (e.g. `/tmp/ccrA` ->
`-tmp-ccrA`). `--resume <sid> --fork-session` only finds a session whose jsonl
lives under the **current** cwd's project dir. So per-task workspaces would each
fail to fork a session primed in a different cwd. The runner closes that gap: on
prime it records both the primed `session_id` **and** the absolute path of the
jsonl it created (under the prime cwd's project dir); before each fork it
**copies** that jsonl into the **task cwd's** project dir (a no-op when the task
cwd equals the prime cwd) so `--resume` finds it, then forks. The
`claude_session_store.py` helpers (`encode_cwd_to_project_dirname`,
`project_dir_for_cwd`, `session_jsonl_path`, `ensure_session_present_in_cwd`)
implement this; the projects root defaults to `~/.claude/projects` and is
overridable via the `CLAUDE_PROJECTS_ROOT` env var or the `projects_root=`
argument to `run_claude_code_task` (tests point it at a tmp dir).

**Registry.** A tiny JSON map
`{chunk_id: {"session_id": <psid>, "source_jsonl": <abs jsonl path>}}` at
`~/.claude_code_cli_runner/reusable_sessions.json` (override via the
`CLAUDE_CODE_CLI_RUNNER_REGISTRY_PATH` env var, or the `registry_path=` argument
to `run_claude_code_task`). Older bare-string `{chunk_id: primed_session_id}`
entries are still read for back-compat. Writes are an atomic same-dir replace
under a lock.

**Inline fallback — always correct, never optional for correctness.** Reuse is
purely an optimization. The chunk is instead **prepended inline** to
`input_content` (chunk blocks first, then the task blocks) whenever:

- `enable_session_reuse` is `False`, OR
- no `reusable_context` is supplied, OR
- the chunk is **not text-only** (contains image/document blocks), OR
- priming/forking raises or errors (e.g. the prime exits non-zero or times out,
  the primed session jsonl is missing so it cannot be relocated, the copy fails,
  or claude rejects `--resume`).

On a reuse failure the task still completes via the inline path, a
`{"runner_note": ...}` record is written to the live log, and the unusable
`chunk_id` is dropped from the registry so it is re-primed next time.
Correctness never depends on reuse succeeding.

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

Run states reflected into `task_run_status.json` are `running`, `paused`, and
`operator_ended` (the last is set only by an `end_and_return` intent).

### Consuming the live log and driving controls

While a run is in progress (e.g. from another thread/process), tail the live log
and write control intents. The package ships the helpers for both directions:

```python
import json, time
from claude_code_cli_runner import live_log_path
from claude_code_cli_runner.live_files import append_control_intent

ws = "/path/to/workspace"

# Read: tail the append-only JSONL live log.
with open(live_log_path(ws), "r", encoding="utf-8") as fh:
    for line in fh:                      # each line: {"received_at": ..., "chunk": {...}}
        record = json.loads(line)
        print(record.get("chunk") or record.get("raw"))

# Write: drive the out-of-band control channel.
append_control_intent(ws, "pause")
append_control_intent(ws, "send_command", command_text="also check the logs")
append_control_intent(ws, "resume")
append_control_intent(ws, "end_and_return")
```

A `pause` blocks the runner's stream-consumption loop until a `resume` (or
`end_and_return`) arrives; `send_command` injects a mid-run user turn over the
process's still-open stdin; `end_and_return` stops the run early and sets
`operator_ended=True` on the result. The same control channel is exposed on the
CLI as `python3 -m claude_code_cli_runner control --workspace /ws --intent ...`.

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

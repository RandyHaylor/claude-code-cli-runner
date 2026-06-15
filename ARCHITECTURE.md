# Architecture

## Design thesis

There is exactly **one** unified, always-streaming execution core:
`run_claude_code_task`. Everything that looks like it might be a different
"mode" — where the task runs (local vs SSH), which face invokes it (library /
CLI / HTTP), what kind of content goes in or comes out (text / image /
document / files) — is expressed as a **parameter or a seam**, never as a
forked code path.

The slogan is **"location is config, not code."** A local subprocess run and a
run on a VM over SSH differ only in the argv that gets built; both feed the
identical streaming loop, produce the identical live-window files, and return
the identical `RunResult` shape.

The package also knows nothing about any particular orchestrator. It has no
task-return envelope, no composed-context assembly, no effort/thinking policy,
and no queue/runner state machine. Those belong to a thin consumer-side adapter
(see "Integration points").

## Module map

| Module | Role |
|---|---|
| `request.py` | The `RunRequest` dataclass + the multimodal input content-block types (`TextBlock` / `ImageBlock` / `DocumentBlock`) + `SshConfig` and the `execution_location` constants. Pure data; validates `execution_location` in `__post_init__`. |
| `content.py` | Serialize input content blocks into the stream-json user message `content` array; build the injected (`send_command`) user message; snapshot the workspace and diff it to list **produced artifacts** (multimodal out). |
| `transports.py` | The transport seam: `build_command_for(request) -> argv`. The ONLY place `execution_location` branches. Local runs `claude` directly; `vm_over_ssh` / `remote_host` wrap the same claude argv in `ssh`. Holds the VM-name→IP DHCP resolution seam. |
| `runner.py` | The single streaming execution core: launch the subprocess, deliver the prompt over stdin, append every chunk to the live log, honour the control channel, reflect run-state, and assemble the `RunResult`. |
| `result.py` | The `RunResult` dataclass + `to_dict()`. Multimodal-aware: assistant text **and** produced artifacts, plus exit/status and the live-log path. |
| `live_files.py` | The on-disk live-window contract: the three file names, the run-state values, the four control intents, and the read/write helpers around them. The load-bearing names live here in one place so every face agrees. |
| `__main__.py` | The CLI face: `run` / `serve` / `control` subcommands. Builds a `RunRequest` from argv and drives the same core. Also the `claude-code-cli-runner` console script. |
| `http_server.py` | The optional streaming HTTP face: `POST /run` runs the core in a background thread, emits a `{"run_started": {run_id, workspace_token}}` line, streams the growing live log back, then a final `{"run_result": {...}}` line. `POST /control` writes a control intent into a run's control channel (run identity = `workspace_token` = workspace_directory). Ships `request_from_json`, `build_streaming_http_server`, `run_streaming_http_server`, the incremental `stream_run` client, the buffered `post_run` client, and `post_control`. |
| `__init__.py` | Public surface: `run_claude_code_task`, `RunRequest`, `RunResult`, the content blocks, and the live-window names/intents/path helpers. |

## Data flow

```
RunRequest(input_content=[...blocks...], workspace, execution_location, ssh, ...)
      |
      |  content.build_user_message()            transports.build_command_for()
      |  -> stream-json user message             -> argv  (local: claude ...;
      v     (content array)                          ssh: ssh ... 'claude ...')
   +--------------------------------------------------------------------------+
   |  runner.run_claude_code_task                                             |
   |                                                                          |
   |   subprocess.Popen(argv, cwd=workspace, stdin/stdout/stderr=PIPE)        |
   |     stdin  <- user message (then stays OPEN for send_command injection)  |
   |     stdout -> for each NDJSON chunk:                                     |
   |                 - drain control channel (pause/resume/send/end)          |
   |                 - append chunk to task_live_log.jsonl  (fsync'd)         |
   |                 - collect assistant text / final result event           |
   |     side-effects -> task_run_status.json   (running/paused/operator_ended)|
   |                  -> reads task_control_channel.jsonl                      |
   +--------------------------------------------------------------------------+
      |
      |  content.list_produced_artifacts(workspace, baseline)   (workspace diff)
      v
RunResult(assistant_text, final_result_event, produced_artifacts,
          exit_code, operator_ended, live_log_path, run_state, workspace_directory)
```

## The transport seam

`build_command_for(run_request)` is the entire location-dispatch surface:

- `local_subprocess` → `build_base_claude_argv` — `claude -p --output-format
  stream-json --include-partial-messages --input-format stream-json --verbose`
  (plus `--model`, `--dangerously-skip-permissions`, and `extra_cli_flags` when
  set). The prompt is **never** a positional arg — it is delivered over stdin —
  so it never lands on a process table.
- `vm_over_ssh` / `remote_host` → `build_ssh_argv` — the same base claude argv,
  shell-quoted, optionally prefixed with `cd <remote_workspace_directory>;`,
  wrapped in `ssh [-i key] [-p port] -o StrictHostKeyChecking=no ... user@host`.
  SSH forwards the host process's stdin straight through, so prompt delivery and
  `send_command` injection work identically — just one hop further.

The **VM/DHCP resolution seam** is `resolve_ssh_host`: an explicit `ssh.host`
wins; otherwise `ssh.vm_name` is resolved via libvirt DHCP leases
(`_vm_ip_from_dhcp_leases`, which shells out to `virsh net-dhcp-leases`). This
is the single real-infrastructure dependency, and it is monkeypatched in tests.
Both SSH locations share one builder; `remote_host` vs `vm_over_ssh` is purely a
labelling distinction at the request level.

## Streaming + control-channel model

The runner reads `process.stdout` line by line. Each non-empty line is a stream
chunk. Before processing a chunk it drains the control channel
(`read_new_control_intents`, which tracks a consumed-line offset over the
append-only `task_control_channel.jsonl`):

- `pause` → set the run state to `paused` and spin in a poll loop until
  `resume` / `end_and_return`.
- `resume` → back to `running`.
- `send_command` → write a fresh stream-json user message to the still-open
  stdin (`build_injected_user_message`), injecting a mid-run turn.
- `end_and_return` → set `operator_ended`, reflect `operator_ended`, and break.

Each chunk is wrapped as `{"received_at": ..., "chunk": {...}}` (or `{"raw":
...}` if it was not valid JSON) and appended to `task_live_log.jsonl`, flushed
and `fsync`'d so an external tailer sees it immediately. The runner also pulls
assistant text out of `assistant` chunks and captures the final `result` event;
when a `result` chunk is seen the loop ends. Run-state values written to
`task_run_status.json` are `running`, `paused`, `operator_ended` — these are
live annotations of a process for a UI, not queue/runner states.

## Multimodal model

**In:** `input_content` is a list of content blocks. `content.serialize_block`
turns each into one entry of the stream-json `content` array:

- `TextBlock` → `{"type": "text", "text": ...}`
- `ImageBlock` → `{"type": "image", "source": {"type": "base64",
  "media_type": ..., "data": <b64>}}`
- `DocumentBlock` → `{"type": "document", "source": {...}}` plus an optional
  `"title"` from `name`.

Image/document blocks carry their payload either inline (`data_base64`) or by
`path` (read + base64-encoded at serialization time).

**Out:** before the run, `snapshot_workspace_files` records the workspace's file
set; after the run, `list_produced_artifacts` diffs against that baseline and
returns the new files (the runner's own live-window files are excluded). So
"output" is text **and** files.

**Adding a new modality is data, not code:** add a block dataclass with a
`block_type` in `request.py`, add a branch in `content.serialize_block` keyed
off that type, and (optionally) a builder in `http_server._BLOCK_BUILDERS` and a
CLI flag. The runner never changes.

**Adding a new transport** is similarly localized: write a
`build_<location>_argv`, register its `execution_location` constant, and add the
branch in `build_command_for`. Everything downstream is untouched.

## The three faces — one core

| Face | Entry point | How it drives the core |
|---|---|---|
| Library | `run_claude_code_task(RunRequest(...))` | Direct call; returns a `RunResult`. |
| CLI | `python3 -m claude_code_cli_runner run\|serve\|control` | Builds a `RunRequest` from argv, calls the core, prints the `RunResult` JSON. `control` writes the control channel; `serve` starts the HTTP face. |
| HTTP | `POST /run` + `POST /control` (`build/run_streaming_http_server`, `stream_run`/`post_run`, `post_control`) | `/run` parses JSON into a `RunRequest`, runs the core on a background thread, emits a `run_started` identity line, streams the live log back incrementally, then a final `{"run_result": {...}}`. `/control` forwards a control intent into the run's control channel keyed by `workspace_token`. |

All three converge on the same `run_claude_code_task`. The faces differ only in
how a `RunRequest` is constructed and how a `RunResult` is surfaced.

## Integration points

A consumer (e.g. an orchestrator) wraps this with a **thin adapter**:

- map its own task/context object → a `RunRequest` (its prompt assembly,
  policy, model choice become block lists and request fields here);
- call `run_claude_code_task`;
- map the `RunResult` → its own result/envelope type;
- optionally tail `task_live_log.jsonl` / write `task_control_channel.jsonl`
  for a live dashboard, using the contract names from `live_files.py`.

This library **deliberately does not** provide: any task-return envelope shape,
composed-context prompt assembly, effort/thinking-to-settings policy, container
lifecycle management, definition-of-done, or queue/runner state machines. Those
are orchestrator concerns and stay in the adapter, keeping this core generic.

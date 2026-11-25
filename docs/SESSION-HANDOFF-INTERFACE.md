# Continue the same task in a fresh session from a handoff document

Status: Claude path built (this repo). Pi path investigated only (section 6).

## 1. The architect's ruling (verbatim)

"the runs arent' blocked' they just use the tool call blocking as a way to get the agent's
attention. the agent then must follow the handoff procedure by writing a clear handoff
document and sendign a request up to the vm runner via it's own tooling (won't be blocked by
the claude hook) to spawn a new session under the same task, give it the handoff doc path,
and update the session id in the task, and have it continue work."

Plus: "ALSO ensure that there's a clean interface so the vm runner can receive and respond to
a generic request like this to suport any agent harness, and also look into how to create it
for the pi harness as well."

## 2. The generic request and response (harness-agnostic)

The request travels on the runner's existing text request channel: a fenced block in the
agent's turn text, recognised by the runner at the turn's stop (the same mechanism the
runner already uses for `unharness-tool` calls). It is NOT a harness tool call, so no
harness hook ever sees it — this is the "own tooling (won't be blocked by the claude hook)".

Request (the agent emits, then ends its turn):

    ```runner-request
    {"request": "continue_this_task_in_a_fresh_session_from_a_handoff_document",
     "handoff_document_path": "/abs/path/to/handoff.md"}
    ```

Runner response:

- Accepted (the document exists and is non-empty): the runner ends the current harness
  session, starts a NEW session of the same harness in the same working directory for the
  same run (same task), and gives it this first message:
  "Continue the task. ... Read the handoff document at <path> first ...".
  The run keeps streaming; nothing goes back to the old session.
- Refused (no such file, empty file, unknown request, no harness support): the runner sends
  the refusal reason back into the SAME session as its next input and the agent goes on
  (it can fix the document and ask again).

Every request and response is written to the live log as a `runner_note`, and the run result
carries `fresh_sessions_started_from_handoff_documents`: one record per accepted request,
`{previous_session_id, new_session_id, handoff_document_path, started_at}`.

Code: `claude_code_cli_runner/fresh_session_from_handoff_document.py` (request shape,
detection, validation, the messages) and the handoff branch in `runner.py`.

## 3. How a harness adapter plugs in

A harness integration opts in by implementing one method:

    start_fresh_session_from_handoff_document(*, current_process, first_message_text,
        run_request, launch_harness_subprocess) -> (new_process, new_session_id_or_None)

It ends the current session's process, launches a new session on its own terms, delivers
`first_message_text` as that session's first input, and returns the process plus the new
session id when it knows it up front (`None` when the harness mints it and reports it on the
stream). A harness without the method gets a refusal response (section 2) — never a crash.

- Claude (built): terminates the process, mints a uuid, launches the normal streaming argv
  with `--session-id <uuid>` (not `--resume`), same settings/hook/model/permissions, and
  writes the first message as a stream-json user message.
- Pi (not built): see section 6.

## 4. How the new session id reaches the Unharness task

The existing path, unchanged: after the handoff the runner's result reports the NEW session
(`harness_session_id` and the final result event's `session_id` both come from the fresh
session). The connector (`work_assignment_connector/adapters.py`
`map_run_result_to_envelope`) puts it on the envelope as `harness_session_id`, and the
supervisor (`live_task_execution_supervisor/supervisor.py`, "PER-TASK SESSION PERSISTENCE")
stores it as the task's `body.worker_session_id`:
- first run of the task: it stores whatever id was reported -> the new id;
- resumed run reporting a different id: it stores the new id. Today that branch also writes
  the system-log line "resume failed for task ... started fresh session ...", which is
  WRONG wording for a handoff (open question 1).

## 5. The attention signal for Claude

- Hook: a `PreToolUse` command hook added by the runner into its existing single inline
  `--settings` argument (`claude_harness.build_base_claude_argv`), merged with the caller's
  `claude_settings_overrides` (any hooks the caller already set are kept).
- Script: `claude_code_cli_runner/context_threshold_handoff_attention_hook.py`. It measures
  the context exactly as the `claude-code-token-count` gate does, by importing that gate's
  own `read_last_usage` from `token_usage_gate.py` (last `message.usage`
  `all_observed_tokens` in the hook payload's `transcript_path`). It does not call the gate's
  `main`, because the gate's deny text tells a human to type an unblock code, and its
  allowance file is per-project, not per-run.
- Gate location: env `CLAUDE_CODE_TOKEN_COUNT_GATE_SCRIPT`, else
  `~/.claude/skills/claude-code-token-count/scripts/token_usage_gate.py`. The dev stack
  deploys only the runner package into the VM today, so `deploy/unharness_dev_stack.sh`
  (`deploy_runner_into_vm_lockstep`) now also copies the gate script to
  `/home/agent-user/.claude/skills/claude-code-token-count/scripts/` in the VM. If the runner
  cannot find the gate at launch it does NOT install the hook and writes a `runner_note`
  saying so (visible, never silent).
- Past the threshold the hook DENIES the tool call (`permissionDecision: deny`) with the
  handoff procedure as the reason: write the handoff document to the given path, then emit
  the `runner-request` block and stop.
- Exemption: the handoff request itself is text, so it is never a tool call and needs no
  exemption. The one tool call the hook lets through past the threshold is a `Write` whose
  `file_path` is exactly this session's handoff document path
  (`<run working directory>/session_handoff_documents/handoff_from_session_<session id>.md`),
  so the agent can write the document. Every other call is denied.
- Only for `execution_location == local_subprocess` (the in-VM serve runs claude as a local
  subprocess). Over `vm_over_ssh`/`remote_host` the hook path would not exist on the far
  side, so the runner writes a `runner_note` and does not install it.

## 6. Threshold (per run) — CHOICE, the architect did not set it

New RunRequest/wire field `fresh_session_handoff_context_token_threshold` (int tokens;
absent = no hook). Unharness computes it per dispatch in the supervisor as
`resume_shared_session_only_when_its_context_used_fraction_is_below` (operational profile:
0.75 for `llm service`, 0.50 for `local llm`) x `model_context_window_tokens` (selected
preset). If either value is not configured, nothing is sent and no hook is installed. No
default preset sets `model_context_window_tokens` today, so out of the box the feature is
off until a preset sets it (open question 2).

## 7. Pi (investigated, not built)

- Where: `claude_code_cli_runner/pi_harness.py` (RPC mode, `pi --mode rpc`), extensions in
  `pi_harness_extensions/*.ts` loaded with `-e` via `PI_EXTENSION_PATHS`.
- Attention signal: Pi extensions get a `tool_call` event "before the tool executes. Can
  block." returning `{ block: true, reason }`, and `ctx.getContextUsage()` "Returns current
  context usage for the active model" (pi-mono `packages/coding-agent/docs/extensions.md`).
  So a small extension `context_threshold_handoff_attention.ts` would do exactly what the
  Claude hook does: past the threshold block every call except a `write` to the handoff
  path, with the same procedure text. Threshold and path would reach it via environment
  variables set by the runner (`run_environment_variables`).
- Request: the same `runner-request` text block; the runner's detection is already
  harness-agnostic (Pi turn text reaches the same stop check).
- New session: `PiHarnessIntegration.start_fresh_session_from_handoff_document` would end the
  RPC process and launch `build_launch_command` WITHOUT `--session` (Pi mints the id; no
  caller-chosen ids), then send the first message as an RPC `prompt` command plus
  `get_state` as `deliver_prompt` already does. It returns `None` for the id.
- Session id reporting: Pi reports the minted id via `get_state` (`data.sessionId`), which the
  Pi normalizer already turns into a uniform `{"type":"session","session_id":...}` chunk.
  The runner core already handles this: when the integration returns `None` for the id, the
  core clears the run's captured session id so the NEXT session chunk becomes the run's
  session (built and harmless for Claude, which returns its id directly).
- Not checked: whether Pi in RPC mode honours a `tool_call` block from an `-e` extension the
  same way as interactive mode, and whether `ctx.getContextUsage()` is populated there. Both
  need a run in the VM before building.

## 8. Open questions for the architect

1. The supervisor's "resume failed" system-log line also fires after a handoff on a resumed
   run. Should the run result's handoff records drive a distinct log line (needs an envelope
   field from the connector and a supervisor branch), or is the existing line acceptable?
2. Where should `model_context_window_tokens` come from for the default presets?
3. Should the handoff document stay in the run directory (current: excluded from the agent's
   git via the runner's bookkeeping excludes) or be stored on the task node?

# Runner MCP integration layer — design

Status: **DESIGN ONLY. Not built.** This documents the layer that lets the runner act
as a generic MCP client over any harness it drives. It is gated behind a precondition
(see "Precondition" below): the standalone Unharness MCP server must be fully working
and tested first.

## Purpose

Make the **runner** a generic **MCP client** that sits OVER the specific harness
integrations, so an agent with **no native tool-calling for these tools** can still use
runner-provided MCP tools (e.g. the Unharness API). The harness stays model-agnostic;
the runner mediates the tool loop. (SoT nd-459/nd-460: runner is the generic MCP client
over harnesses.)

This is the same shape the classification path already uses for ONE turn (agent emits a
structured thing → code parses + acts), generalized into a multi-turn loop.

## The three moving parts

1. **The standalone Unharness MCP server** (already built, `tools/unharness-mcp-server`):
   a stateless, stdio, industry-standard MCP server wrapping the Unharness API. The
   runner spawns it and speaks standard MCP (`initialize` → `tools/list` → `tools/call`)
   to it. Reused as-is — the runner is just one of its MCP clients. (SoT nd-458/nd-470.)

2. **A translation table (the "digestible" surface).** The agent is prompted with a
   MORE DIGESTIBLE toolset (simpler names/shape — the "b" surface) than the raw MCP tool
   schema. The runner-MCP layer holds a table mapping each digestible call to the real
   standard MCP `tools/call` ("a"). So the agent speaks the easy shape; the wire speaks
   the standard. This mirrors the existing generic→harness tool-name translation
   (`harness_tool_name_translation` / `tool_name_conversion_reference.json`), one layer
   up. It is ALSO where a role/subset can expose fewer tools than the server offers.
   (SoT nd-461/nd-473: runner layer translates digestible calls to standard MCP.)

3. **The stop-gated tool loop** (the core new behavior; SoT nd-472/nd-474/nd-476):

   - The runner already consumes the `claude -p` stream chunk by chunk. The
     **`result` stream event is the turn's STOP** (`runner.py`, where `final_result_event`
     is set) — the terminal signal Unharness reacts to. We use THIS existing stop; we do
     NOT invent another. **StopFailure is not detected now — a stop is a stop.**
   - At the stop, the layer inspects the just-completed turn's text for a **detected
     digestible-shape tool call** (regex/schema recognition).
     - **If a tool call is found:** the runner
       1. does NOT let the stop propagate up to Unharness,
       2. translates the digestible call → standard MCP `tools/call`, forwards it to the
          MCP server, gets the result (even empty/error becomes a message, e.g.
          `"unharness-api command submitted, response: ..."`),
       3. injects the result back to the agent as the next input over the EXISTING stdin
          user-message path (`write_stream_json_message(build_user_message(...))`), and
       4. continues the loop — the agent keeps going.
     - **If no tool call is found:** the stop propagates up as it does today; the run
       ends and Unharness reacts to the stop.
   - **The stream still flows up to Unharness continuously as normal.** Only the terminal
     `stop` is gated. All other text (including text alongside a tool call) is preserved
     in conversation history and in the upward stream, unchanged. Tool round-trips are
     invisible to Unharness except as ordinary streamed text.

## What is NOT in scope here (explicitly deferred)

- **StopFailure detection / typed API-error handling** — not now.
- **Re-engage-on-idle behavior** — already implicit in the heartbeat paradigm we migrated
  to; NOT this layer's concern (SoT nd-476, per the architect).
- **Native harness MCP config** (e.g. `--mcp-config`) — we do NOT rely on the harness's
  own tool-calling; the runner owns the loop so it stays harness-agnostic (SoT nd-438
  deprecated the "no MCP" stance in favor of this runner-mediated MCP).

## Verified seams in the current runner (read-only, this session)

- Stop point: `chunk_type == "result"` in the stream loop sets `final_result_event`.
- Next-input injection already exists: `write_stream_json_message(build_user_message(...))`
  writes a stream-json user message to the harness stdin (how the initial prompt and the
  permission/resume path deliver input).
- A `--settings` injection point already exists in `transports.py` (currently pinning the
  permission `defaultMode`) — a seam if hook/settings injection is ever needed.
- Run-state reflection (`_reflect(status_path, RUN_STATE_*)`) already surfaces states
  (incl. idle-killed) upward — where a stop/idle signal would ride.

## NOT YET confirmed (must verify before coding)

- Whether, after a `result` event, the current loop is single-shot (terminates) or can
  already continue another turn on the SAME session/stdin without a full re-spawn. This
  determines whether the loop injects-and-continues in place or resumes the session.
- The exact digestible-call **shape** the agent emits and the prompt that teaches it
  (author with the architect; capture verbatim). The translation table keys off this.

## Precondition (sequencing, SoT nd-477)

The standalone Unharness MCP server must be **fully working and tested** BEFORE this
integration is built. "Fully working" here means beyond the current fake-Unharness tests:
exercised against a REAL running Unharness app, and with whatever tool surface the real
ingest/assistant work actually needs (today it exposes a minimal read set + one
reference-anchored task-creation tool).

## First testable slice (when the precondition is met)

Detector + translation + single-call loop against a STUBBED harness (no live Claude):
prove "a turn containing a digestible tool call loops and feeds the result back" and "a
turn with no tool call ends (stop propagates)". Then layer in the real stdin injection
and a real MCP-server round-trip.

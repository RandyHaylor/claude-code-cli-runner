"""The STANDARD harness-activity-event vocabulary for live-log records.

Every live-log record the runner writes carries an ``activity`` facet in this
neutral, harness-agnostic vocabulary (approved standard, raw-780/781). The raw
harness chunk always rides alongside untouched (``chunk``) for troubleshooting;
renderers read ONLY ``activity`` and never need harness-specific knowledge.

Kinds:
  - ``assistant_text``      {text}
  - ``reasoning_text``      {text}                       (log-only thinking)
  - ``tool_activity``       {tool_name, status, input_summary, output_summary}
  - ``turn_usage``          {token_usage}
  - ``run_result``          {final_text}
  - ``runner_note``         {text}
  - ``permission_request``  {request_id, tool_name}
  - ``permission_resolved`` {request_id, behavior}

``derive_harness_activity_event`` maps ONE parsed harness chunk (claude-shaped
or an opencode passthrough event) to an activity dict, or None when the chunk
carries nothing an operator needs rendered (protocol bookkeeping, step
boundaries)."""

from __future__ import annotations

_SUMMARY_CHARACTER_LIMIT = 400


def _bounded_summary(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            import json

            value = json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
    return value[:_SUMMARY_CHARACTER_LIMIT]


def derive_harness_activity_event(chunk) -> "dict | None":
    """Map one parsed harness chunk to a standard activity event (or None)."""
    if not isinstance(chunk, dict):
        return None
    chunk_type = chunk.get("type")

    # --- claude-shaped chunks (also produced by the opencode translator) -----
    if chunk_type == "assistant":
        message = chunk.get("message") or {}
        blocks = message.get("content") or []
        text_parts = [
            block.get("text", "") for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        tool_blocks = [
            block for block in blocks
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        # TEXT dedup: a real claude run streams its text as partial deltas
        # (handled below) and then repeats it in the full assistant message —
        # emit assistant_text here ONLY for translator-produced chunks (marked
        # with ``opencode_event``), whose text arrives exactly once.
        if chunk.get("opencode_event") and text_parts and "".join(text_parts):
            return {"kind": "assistant_text", "text": "".join(text_parts)}
        if tool_blocks:
            block = tool_blocks[0]
            return {
                "kind": "tool_activity",
                "tool_name": block.get("name") or "(tool)",
                "status": "requested",
                "input_summary": _bounded_summary(block.get("input")),
                "output_summary": "",
            }
        return None
    if chunk_type == "stream_event":
        event = chunk.get("event") or {}
        if not isinstance(event, dict):
            return None
        if event.get("type") == "message_delta" and event.get("usage"):
            return {"kind": "turn_usage", "token_usage": dict(event["usage"])}
        # claude's streamed partial text: the once-only text source for a
        # real claude run (see the assistant-message dedup note above).
        delta = event.get("delta") or {}
        if (
            event.get("type") == "content_block_delta"
            and isinstance(delta, dict)
            and delta.get("type") == "text_delta"
            and delta.get("text")
        ):
            return {"kind": "assistant_text", "text": delta["text"]}
        return None
    if chunk_type == "result":
        return {"kind": "run_result", "final_text": chunk.get("result") or ""}

    # --- opencode passthrough events -----------------------------------------
    part = chunk.get("part") or {}
    if chunk_type == "reasoning":
        text = part.get("text")
        if isinstance(text, str) and text:
            return {"kind": "reasoning_text", "text": text}
        return None
    if chunk_type == "tool_use":
        state = part.get("state") or {}
        return {
            "kind": "tool_activity",
            "tool_name": part.get("tool") or "(tool)",
            "status": state.get("status") or "unknown",
            "input_summary": _bounded_summary(state.get("input")),
            "output_summary": _bounded_summary(state.get("output")),
        }
    if chunk_type == "text":
        text = part.get("text")
        if isinstance(text, str) and text:
            return {"kind": "assistant_text", "text": text}
        return None

    return None

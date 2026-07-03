"""Translate opencode ``--format json`` events into the claude-style chunk
shapes the streaming runner already consumes.

The runner's streaming core understands three claude chunk shapes:
  - ``{"type": "assistant", "message": {"content": [{"type": "text", ...}]}}``
    -> rendered assistant text
  - ``{"type": "stream_event", "event": {"type": "message_delta",
    "usage": {...}}}`` -> cumulative token accounting
  - ``{"type": "result", "result": "<final text>", "usage": {...}}``
    -> terminal event (ends the streaming loop)

opencode emits (verified against opencode 1.17.13 against a local ollama
provider):
  - ``{"type": "text", "part": {"text": ...}}`` -> assistant text
  - ``{"type": "step_finish", "part": {"reason": "stop"|..., "tokens":
    {"input": N, "output": N, "reasoning": N,
    "cache": {"write": N, "read": N}}}}`` -> per-step usage; ``reason ==
    "stop"`` is the run's final step
  - other events (step_start, tool, ...) -> informational only

This translator is a small stateful object because the final claude-style
``result`` chunk must carry the run's full assistant text, which opencode
spreads over earlier ``text`` events.
"""

from __future__ import annotations


def map_opencode_token_block_to_usage(tokens: dict) -> dict:
    """Map opencode's step token block onto claude-style usage field names."""
    cache = tokens.get("cache") or {}
    return {
        "input_tokens": int(tokens.get("input") or 0),
        "output_tokens": int(tokens.get("output") or 0),
        "cache_creation_input_tokens": int(cache.get("write") or 0),
        "cache_read_input_tokens": int(cache.get("read") or 0),
    }


class OpencodeEventToClaudeChunkTranslator:
    """Feed raw parsed opencode events in; get claude-style chunk dicts out.

    ``translate(event) -> list[dict]``: zero or more claude-style chunks for
    the runner to log/render/account. Unrecognized events pass through
    unchanged (they land in the live log as-is; the runner ignores unknown
    chunk types harmlessly).
    """

    def __init__(self) -> None:
        self._collected_assistant_text_parts: "list[str]" = []
        self._collected_reasoning_text_parts: "list[str]" = []
        self.last_seen_session_id: "str | None" = None

    def translate(self, opencode_event: dict) -> "list[dict]":
        if not isinstance(opencode_event, dict):
            return []
        session_id = opencode_event.get("sessionID")
        if isinstance(session_id, str) and session_id:
            self.last_seen_session_id = session_id
        event_type = opencode_event.get("type")
        part = opencode_event.get("part") or {}

        if event_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                return [opencode_event]
            self._collected_assistant_text_parts.append(text)
            return [
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": text}]},
                    "opencode_event": opencode_event,
                }
            ]

        if event_type == "reasoning":
            reasoning_text = part.get("text")
            if not isinstance(reasoning_text, str):
                return [opencode_event]
            self._collected_reasoning_text_parts.append(reasoning_text)
            # Logged (visible to the operator) but NOT rendered as assistant
            # text — reasoning is the model's working, not its reply.
            return [opencode_event]

        if event_type == "step_finish":
            usage = map_opencode_token_block_to_usage(part.get("tokens") or {})
            usage_chunk = {
                "type": "stream_event",
                "event": {"type": "message_delta", "usage": usage},
                "opencode_event": opencode_event,
            }
            if part.get("reason") == "stop":
                final_text = "".join(self._collected_assistant_text_parts)
                result_chunk = {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": final_text,
                    "usage": usage,
                    "session_id": self.last_seen_session_id,
                    "harness": "opencode_cli",
                }
                if not final_text and self._collected_reasoning_text_parts:
                    # Some local models (observed: ollama gemma4:e4b) put their
                    # ENTIRE answer in reasoning and stop. The model's own
                    # reasoning text is then the only reply there is — surface
                    # it as the result (flagged) rather than returning a blank.
                    result_chunk["result"] = "".join(
                        self._collected_reasoning_text_parts
                    )
                    result_chunk["result_text_source"] = "reasoning_only"
                return [usage_chunk, result_chunk]
            return [usage_chunk]

        return [opencode_event]

#!/usr/bin/env python3
"""A STUB of a streaming `claude -p` for the fresh-session-from-handoff tests. NEVER the
real CLI.

Launched WITHOUT --session-id (the first session): it reads the prompt, optionally writes a
handoff document (STUB_HANDOFF_DOCUMENT_PATH, when STUB_WRITE_HANDOFF_DOCUMENT=1), emits a
turn whose text carries the runner-request block for that path, and stops (result event).
If the runner then sends a message back into this session (a refusal), it answers with
"FIRST_SESSION_GOT::<message>" and a final result.

Launched WITH --session-id <id> (the fresh session): it reads its first message and answers
"FRESH_SESSION_GOT::<message>", reporting <id> as its session id.
"""

import json
import os
import sys
import time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_next_user_message_text():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("type") != "user":
            continue
        content = (message.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        return " ".join(
            block.get("text", "") for block in content or [] if isinstance(block, dict)
        )
    return None


def answer(session_id, text):
    emit({"type": "assistant", "session_id": session_id,
          "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})
    time.sleep(0.02)
    emit({"type": "result", "subtype": "success", "is_error": False,
          "session_id": session_id, "result": text})


def main():
    if "--session-id" in sys.argv:
        session_id = sys.argv[sys.argv.index("--session-id") + 1]
        emit({"type": "system", "subtype": "init", "session_id": session_id})
        first_message = read_next_user_message_text() or ""
        answer(session_id, "FRESH_SESSION_GOT::" + first_message)
        return

    session_id = "stub-first-session"
    emit({"type": "system", "subtype": "init", "session_id": session_id})
    read_next_user_message_text()
    handoff_document_path = os.environ["STUB_HANDOFF_DOCUMENT_PATH"]
    if os.environ.get("STUB_WRITE_HANDOFF_DOCUMENT") == "1":
        with open(handoff_document_path, "w", encoding="utf-8") as handle:
            handle.write("# Handoff\nDone: step 1. Next: step 2.\n")
    request_text = (
        "Context is full; handing off.\n```runner-request\n"
        + json.dumps({
            "request": "continue_this_task_in_a_fresh_session_from_a_handoff_document",
            "handoff_document_path": handoff_document_path,
        })
        + "\n```\n"
    )
    answer(session_id, request_text)
    followup = read_next_user_message_text()
    if followup is not None:
        answer(session_id, "FIRST_SESSION_GOT::" + followup)
        return
    time.sleep(30)


if __name__ == "__main__":
    main()

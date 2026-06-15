#!/usr/bin/env python3
"""A STUB of a streaming `claude -p` process for tests. NEVER the real CLI.

Emits realistic stream-json NDJSON on stdout (system/init, content deltas, an
assistant message, and a final result event), one line at a time with a small
delay so a control channel can be polled between lines. Reads stdin (the
--input-format stream-json channel) on a background thread and ECHOES any
injected user message back into its own stdout, so tests can assert that a
prompt / send_command actually reached the process.

It can also write a workspace artifact (to exercise multimodal OUTPUT capture).

Env knobs:
  STUB_LINE_DELAY_SECONDS  per-line delay (default 0.02)
  STUB_NUM_DELTAS          number of content deltas (default 3)
  STUB_RESULT_TEXT         the final result text (default "stub assistant text")
  STUB_RUN_FOREVER         if "1", emit heartbeats forever, never a result
  STUB_WRITE_ARTIFACT      if set to a filename, write it under cwd before result
  STUB_ARTIFACT_CONTENT    contents for that artifact (default "stub artifact")
"""

import json
import os
import sys
import threading
import time

LINE_DELAY = float(os.environ.get("STUB_LINE_DELAY_SECONDS", "0.02"))
NUM_DELTAS = int(os.environ.get("STUB_NUM_DELTAS", "3"))
RUN_FOREVER = os.environ.get("STUB_RUN_FOREVER", "0") == "1"
RESULT_TEXT = os.environ.get("STUB_RESULT_TEXT", "stub assistant text")
WRITE_ARTIFACT = os.environ.get("STUB_WRITE_ARTIFACT")
ARTIFACT_CONTENT = os.environ.get("STUB_ARTIFACT_CONTENT", "stub artifact")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def stdin_echo_loop():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        emit({"type": "injected_echo", "received": message})


def main():
    threading.Thread(target=stdin_echo_loop, daemon=True).start()

    emit({"type": "system", "subtype": "init", "session_id": "stub-session"})
    time.sleep(LINE_DELAY)
    for index in range(NUM_DELTAS):
        emit(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "chunk%d " % index},
                },
            }
        )
        time.sleep(LINE_DELAY)

    if RUN_FOREVER:
        while True:
            emit({"type": "system", "subtype": "status", "status": "working"})
            time.sleep(LINE_DELAY)

    if WRITE_ARTIFACT:
        with open(os.path.join(os.getcwd(), WRITE_ARTIFACT), "w", encoding="utf-8") as handle:
            handle.write(ARTIFACT_CONTENT)

    emit(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": RESULT_TEXT}]},
        }
    )
    time.sleep(LINE_DELAY)
    emit({"type": "result", "subtype": "success", "is_error": False, "result": RESULT_TEXT})


if __name__ == "__main__":
    main()

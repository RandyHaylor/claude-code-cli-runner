"""A stub opencode CLI: reads the prompt from stdin until EOF (as real
`opencode run` does when no positional message is given), then emits
opencode-shaped `--format json` events on stdout and exits 0."""

import json
import sys


def main() -> int:
    prompt_text = sys.stdin.read()
    session_id = "ses_stub_opencode_0001"
    events = [
        {
            "type": "step_start",
            "sessionID": session_id,
            "part": {"type": "step-start"},
        },
        {
            "type": "text",
            "sessionID": session_id,
            "part": {"type": "text", "text": "stub saw: " + prompt_text.strip()},
        },
        {
            "type": "step_finish",
            "sessionID": session_id,
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "tokens": {
                    "total": 110,
                    "input": 100,
                    "output": 10,
                    "reasoning": 0,
                    "cache": {"write": 7, "read": 3},
                },
            },
        },
    ]
    for event in events:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Contract checks for the protocol-v2 execution-control state machine."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "scene" / "debugger" / "debug_session_v2.h"
SOURCE = ROOT / "scene" / "debugger" / "debug_session_v2.cpp"


def require(text: str, pattern: str, description: str) -> None:
    if re.search(pattern, text, re.MULTILINE) is None:
        raise AssertionError(f"missing {description}: /{pattern}/")


def main() -> int:
    if not HEADER.exists() or not SOURCE.exists():
        print("debug session v2 model contract: FAIL: state-machine files are missing")
        return 1

    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    combined = header + "\n" + source

    requirements = {
        r"ACTION_PAUSE": "pause action",
        r"ACTION_CONTINUE": "continue action",
        r"ACTION_STEP": "step action",
        r"ACTION_NEXT": "next action",
        r"ACTION_STEP_OUT": "step-out action",
        r'p_message\.type == "step_out"': "step-out command mapping",
        r"duplicate": "duplicate request outcome",
        r"stale_stop_id": "stale stop rejection",
        r"out_of_order_request_id": "out-of-order request rejection",
        r"heartbeat_expired": "heartbeat timeout check",
        r"control_lost": "control-disconnect transition",
        r"STATE_RESUME_REQUESTED": "resume-requested transition",
    }
    try:
        for pattern, description in requirements.items():
            require(combined, pattern, description)
    except AssertionError as error:
        print(f"debug session v2 model contract: FAIL: {error}")
        return 1

    print("debug session v2 model contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

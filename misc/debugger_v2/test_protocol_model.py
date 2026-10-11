#!/usr/bin/env python3

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "scene" / "debugger" / "debug_protocol_v2.h"
SOURCE = ROOT / "scene" / "debugger" / "debug_protocol_v2.cpp"


def require(text: str, pattern: str, description: str) -> None:
    if re.search(pattern, text, re.MULTILINE) is None:
        raise AssertionError(f"missing {description}: {pattern}")


def main() -> int:
    if not HEADER.is_file():
        raise AssertionError(f"missing protocol header: {HEADER}")
    if not SOURCE.is_file():
        raise AssertionError(f"missing protocol source: {SOURCE}")

    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")

    require(header, r'class DebugProtocolV2\s*\{', "protocol class")
    require(header, r'MAJOR\s*=\s*2', "major version")
    require(header, r'MINOR\s*=\s*0', "minor version")

    envelope_fields = [
        "session_id",
        "generation",
        "channel",
        "sequence",
        "request_id",
        "stop_id",
        "type",
        "payload",
    ]
    for field in envelope_fields:
        require(header, rf'\b{field}\b', f"envelope field {field}")

    states = re.findall(r'STATE_[A-Z_]+', header)
    required_states = {
        "STATE_DISCONNECTED",
        "STATE_CONNECTING",
        "STATE_RUNNING",
        "STATE_PAUSE_REQUESTED",
        "STATE_PAUSED",
        "STATE_RESUME_REQUESTED",
    }
    if not required_states.issubset(set(states)):
        missing = sorted(required_states - set(states))
        raise AssertionError(f"missing states: {missing}")
    if len(states) != len(set(states)):
        raise AssertionError("state names must be unique")

    require(source, r'DebugProtocolV2::encode', "envelope encoder")
    require(source, r'DebugProtocolV2::decode', "envelope decoder")
    require(source, r'GDDBG2', "protocol magic")

    print("debug protocol v2 model contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"debug protocol v2 model contract: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)

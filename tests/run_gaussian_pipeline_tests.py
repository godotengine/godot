#!/usr/bin/env python3
"""Legacy wrapper for the Gaussian pipeline smoke tests.

The dedicated pipeline driver previously lived in this module.  The test
orchestration has since been consolidated into `tests/ci/run_baseline_qa.py`.
This wrapper forwards to that runner while pinning the `pipeline` category so
existing automation hooks keep working.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI_RUNNER = ROOT / "tests" / "ci" / "run_baseline_qa.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gaussian pipeline smoke-test wrapper.")
    parser.add_argument("--godot", help="Path to the Godot binary to use.")
    parser.add_argument("--quick", action="store_true", help="Retained for compatibility; ignored.")
    parser.add_argument(
        "--category",
        choices=["ply", "pipeline", "sorting", "runtime"],
        default="pipeline",
        help="Pipeline tests delegate to the central runner (default: pipeline).",
    )
    args = parser.parse_args(argv)

    cmd = [sys.executable, str(CI_RUNNER), "--category", args.category]
    if args.godot:
        os.environ['GODOT_BINARY'] = args.godot
        cmd.extend(["--godot", args.godot])

    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

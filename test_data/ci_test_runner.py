#!/usr/bin/env python3
"""Compatibility wrapper for the legacy CI test runner entry point.

Historically the Gaussian splatting repo exposed `test_data/ci_test_runner.py`
as the way to invoke automated checks.  The authoritative implementation now
lives in `tests/ci/run_baseline_qa.py`; this wrapper simply forwards arguments
so existing scripts keep working while we converge on a single entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_RUNNER = ROOT / "tests" / "ci" / "run_baseline_qa.py"


def build_command(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [sys.executable, str(CI_RUNNER)]
    if args.godot:
        cmd.extend(["--godot", args.godot])
    if args.quick:
        cmd.append("--quick")
    if args.category:
        cmd.extend(["--category", args.category])
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CI runner wrapper (delegates to tests/ci/run_baseline_qa.py).")
    parser.add_argument("--godot", help="Path to the Godot binary to use.")
    parser.add_argument("--docker", action="store_true", help="Legacy flag; enables CI environment defaults.")
    parser.add_argument("--quick", action="store_true", help="Run the trimmed quick-check subset.")
    parser.add_argument(
        "--category",
        choices=["ply", "pipeline", "sorting", "runtime", "module", "qa"],
        help="Run only a specific test category.",
    )
    args = parser.parse_args(argv)

    if args.docker:
        os.environ.setdefault("CI", "true")

    cmd = build_command(args)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

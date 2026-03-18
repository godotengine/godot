#!/usr/bin/env python3
"""Modern compatibility wrapper for the multi-version CI runner.

This keeps the original command-line surface (`--preferred-version`, `--quick`,
etc.) but forwards execution to the single consolidated runner
`tests/ci/run_baseline_qa.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_RUNNER = ROOT / "tests" / "ci" / "run_baseline_qa.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-version CI runner wrapper (delegates to tests/ci/run_baseline_qa.py)."
    )
    parser.add_argument("--godot", help="Path to the Godot binary to use.")
    parser.add_argument("--preferred-version", help="Retained for compatibility; no longer used.")
    parser.add_argument("--quick", action="store_true", help="Run the trimmed quick-check subset.")
    parser.add_argument(
        "--category",
        choices=["ply", "pipeline", "sorting", "runtime", "module", "qa"],
        help="Run only a specific test category.",
    )
    args = parser.parse_args(argv)

    if args.preferred_version:
        print(f"ℹ️  Preferred version '{args.preferred_version}' is ignored; tests rely on the provided Godot binary.")

    cmd = [sys.executable, str(CI_RUNNER)]
    if args.godot:
        os.environ['GODOT_BINARY'] = args.godot
        cmd.extend(["--godot", args.godot])
    if args.quick:
        cmd.append("--quick")
    if args.category:
        cmd.extend(["--category", args.category])

    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compatibility wrapper for the Gaussian splatting test suite runner.

The project now uses `tests/ci/run_baseline_qa.py` as the single source of truth
for orchestrating automated checks.  This wrapper exists so legacy tooling and
documentation that still refers to `test_data/run_all_tests.py` keep working.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_RUNNER = ROOT / "tests" / "ci" / "run_baseline_qa.py"


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    cmd = [sys.executable, str(CI_RUNNER), *args]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

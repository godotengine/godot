#!/usr/bin/env python3
"""Run Gaussian shader contract validation from the canonical module path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "modules" / "gaussian_splatting" / "shaders" / "compile_shaders.py"


def main(argv: list[str]) -> int:
    if not SCRIPT_PATH.is_file():
        print(f"Shader validation script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1

    command = [sys.executable, str(SCRIPT_PATH), *argv]
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

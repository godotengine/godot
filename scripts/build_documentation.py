#!/usr/bin/env python3
"""Orchestrate documentation generation tasks for Godot Gaussian Splatting."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def run(cmd: list[str], description: str) -> int:
    print(f"[docs] {description}: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except FileNotFoundError:
        print(f"[docs] Command not found: {cmd[0]}")
        return 1


def run_doxygen(config: Path) -> int:
    if shutil.which("doxygen") is None:
        print("[docs] Skipping Doxygen generation because 'doxygen' is not installed.")
        return 0
    return run(["doxygen", str(config)], "Running Doxygen")


def run_python(script: Path, *args: str) -> int:
    return run([sys.executable, str(script), *args], f"Running {script.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build documentation artifacts.")
    parser.add_argument("--doxygen", action="store_true", help="Generate C++ API docs via Doxygen.")
    parser.add_argument("--gdscript", action="store_true", help="Extract GDScript documentation.")
    parser.add_argument("--shaders", action="store_true", help="Generate shader documentation.")
    parser.add_argument("--performance", action="store_true", help="Build performance graphs.")
    parser.add_argument("--compatibility", action="store_true", help="Update compatibility matrix.")
    parser.add_argument("--project-settings", action="store_true", help="Regenerate project settings reference.")
    parser.add_argument("--report", action="store_true", help="Refresh documentation coverage reports.")
    parser.add_argument("--all", action="store_true", help="Run all documentation tasks.")
    args = parser.parse_args()

    if args.all:
        args.doxygen = args.gdscript = args.shaders = True
        args.performance = args.compatibility = args.project_settings = args.report = True

    exit_code = 0

    if args.doxygen:
        config = DOCS / "Doxyfile"
        exit_code |= run_doxygen(config)

    if args.gdscript:
        exit_code |= run_python(ROOT / "scripts" / "extract_gdscript_docs.py")

    if args.shaders:
        exit_code |= run_python(ROOT / "scripts" / "generate_shader_docs.py")

    if args.performance:
        exit_code |= run_python(ROOT / "scripts" / "generate_performance_graphs.py")

    if args.compatibility:
        exit_code |= run_python(ROOT / "scripts" / "update_compatibility_matrix.py")

    if args.project_settings:
        exit_code |= run_python(ROOT / "scripts" / "generate_project_settings_reference.py")

    if args.report:
        exit_code |= run_python(ROOT / "scripts" / "documentation_audit.py")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

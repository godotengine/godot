#!/usr/bin/env python3
"""Refactor phase workflow runner for the Gaussian renderer cleanup.

This script automates the repeatable parts of the phased refactor workflow:
- local diff hygiene
- local guard-only checks
- optional architecture pack regeneration
- generation of a native Windows build/test batch helper
- local checkpoint commits

It does not attempt to automate the refactor implementation itself.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_RUNNER = ROOT / "ci" / "scripts" / "run_refactor_phase_windows.bat"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    description: str
    major: bool = False


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("1b.1-closeout", "Finish remaining batched provider/view migration under caveat.", major=True),
    PhaseSpec("1b.2a", "Move provider-based production writers onto explicit mutation access.", major=True),
    PhaseSpec("1b.2b", "Migrate painterly direct-facade access without redesign drift.", major=True),
    PhaseSpec("1b.3", "Split debug/tooling into query view + command sink.", major=True),
    PhaseSpec("1b.4", "Replace tests that mutate internals directly with narrow test hooks.", major=True),
    PhaseSpec("1b.5", "Remove mutable-from-const provider contracts after consumer migration.", major=True),
    PhaseSpec("2-3", "Composition root cleanup coupled with sorting seam removal.", major=True),
    PhaseSpec("4", "Narrow orchestrator dependencies after seam cleanup.", major=True),
    PhaseSpec("5", "Lock down remaining mutable renderer-state access.", major=True),
    PhaseSpec("6", "Optional thin-wrapper cleanup.", major=False),
)

PHASE_MAP = {phase.name: phase for phase in PHASES}


def run_cmd(args: Sequence[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    printable = " ".join(args)
    print(f"$ {printable}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(args, cwd=cwd, env=merged_env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def to_windows_path(path: Path) -> str:
    raw = str(path.resolve())
    if raw.startswith("/mnt/") and len(raw) > 6 and raw[6] == "/":
        drive = raw[5].upper()
        suffix = raw[7:].replace("/", "\\")
        return f"{drive}:\\{suffix}"
    return raw.replace("/", "\\")


def require_phase(name: str) -> PhaseSpec:
    phase = PHASE_MAP.get(name)
    if phase is None:
        known = ", ".join(phase.name for phase in PHASES)
        raise SystemExit(f"Unknown phase '{name}'. Known phases: {known}")
    return phase


def cmd_list(_: argparse.Namespace) -> None:
    print("Refactor phases:")
    for phase in PHASES:
        major = " major" if phase.major else ""
        print(f"- {phase.name}:{major} {phase.description}")


def cmd_local_checks(args: argparse.Namespace) -> None:
    phase = require_phase(args.phase)
    print(f"Running local checks for phase {phase.name}")
    run_cmd(["git", "diff", "--check"], cwd=ROOT)
    if args.guard_only:
        run_cmd(["python3", "tests/ci/run_module_tests.py", "--guard-only"], cwd=ROOT)
    regen_architecture = args.regen_architecture
    if regen_architecture is None:
        regen_architecture = phase.major
    if regen_architecture:
        run_cmd(["python3", "scripts/generate_architecture_diagrams.py"], cwd=ROOT)


def build_windows_runner_contents(phase: PhaseSpec, root: Path) -> str:
    root_win = to_windows_path(root)
    git_dir = to_windows_path(root.parent / "godotgs-clean" / ".git" / "worktrees" / root.name)
    lines = [
        "@echo off",
        "setlocal",
        f'set "PHASE={phase.name}"',
        f'set "ROOT={root_win}"',
        f'set "GIT_DIR={git_dir}"',
        f'set "GIT_WORK_TREE={root_win}"',
        "echo ========================================",
        "echo Gaussian Renderer Refactor Phase Runner",
        "echo Phase: %PHASE%",
        "echo Root : %ROOT%",
        "echo ========================================",
        "cd /d %ROOT%",
        "if errorlevel 1 exit /b %errorlevel%",
        "python tests\\ci\\run_module_tests.py --guard-only",
        "if errorlevel 1 exit /b %errorlevel%",
        "scons platform=windows target=editor dev_build=yes tests=yes module_gaussian_splatting_enabled=yes -j%NUMBER_OF_PROCESSORS%",
        "if errorlevel 1 exit /b %errorlevel%",
        "python tests\\ci\\run_module_tests.py --skip-render-guards",
        "if errorlevel 1 exit /b %errorlevel%",
        "echo Phase %PHASE% verification complete.",
        "endlocal & exit /b 0",
    ]
    return "\r\n".join(lines) + "\r\n"


def cmd_write_windows_runner(args: argparse.Namespace) -> None:
    phase = require_phase(args.phase)
    output = Path(args.output).resolve() if args.output else DEFAULT_WINDOWS_RUNNER
    contents = build_windows_runner_contents(phase, ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents, encoding="utf-8", newline="")
    print(f"Wrote Windows runner to {output}")


def cmd_checkpoint_commit(args: argparse.Namespace) -> None:
    require_phase(args.phase)
    if not args.message:
        raise SystemExit("--message is required for checkpoint-commit")
    if not args.stage_all_tracked and not args.path:
        raise SystemExit("Specify --stage-all-tracked and/or one or more --path entries.")
    if args.stage_all_tracked:
        run_cmd(["git", "add", "-u"], cwd=ROOT)
    for rel_path in args.path:
        run_cmd(["git", "add", "--", rel_path], cwd=ROOT)
    commit_args = ["git", "commit", "-m", args.message]
    run_cmd(commit_args, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List large remaining refactor phases.")
    list_cmd.set_defaults(func=cmd_list)

    checks = sub.add_parser("local-checks", help="Run local diff/guard/architecture checks for a phase.")
    checks.add_argument("--phase", required=True)
    checks.add_argument("--guard-only", action="store_true", default=True)
    checks.add_argument("--regen-architecture", action=argparse.BooleanOptionalAction, default=None)
    checks.set_defaults(func=cmd_local_checks)

    win = sub.add_parser("write-windows-runner", help="Generate a native Windows build/test helper batch file.")
    win.add_argument("--phase", required=True)
    win.add_argument("--output")
    win.set_defaults(func=cmd_write_windows_runner)

    commit = sub.add_parser("checkpoint-commit", help="Create a local checkpoint commit after a verified phase.")
    commit.add_argument("--phase", required=True)
    commit.add_argument("--message", required=True)
    commit.add_argument("--stage-all-tracked", action="store_true")
    commit.add_argument("--path", action="append", default=[], help="Additional file or directory to stage.")
    commit.set_defaults(func=cmd_checkpoint_commit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

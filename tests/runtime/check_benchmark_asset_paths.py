#!/usr/bin/env python3
"""Guard against hardcoded benchmark/synthetic PLY paths in scenes/scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT / "tests" / "examples" / "godot" / "test_project"

SCAN_ROOTS = (
    PROJECT_ROOT / "scenes",
    PROJECT_ROOT / "scripts",
)

SCAN_SUFFIXES = {".gd", ".tscn"}
TARGET_NAME_TOKENS = ("benchmark", "synthetic")
HARDCODED_PLY_RE = re.compile(r"res://tests/fixtures/[A-Za-z0-9_\-]+\.ply")


def _iter_candidate_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SCAN_SUFFIXES:
                continue
            lowered = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/").lower()
            if not any(token in lowered for token in TARGET_NAME_TOKENS):
                continue
            out.append(path)
    return sorted(out)


def main() -> int:
    violations: list[str] = []
    for path in _iter_candidate_files():
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            violations.append(f"{rel_path}: could not read file ({exc})")
            continue
        for idx, line in enumerate(lines, start=1):
            match = HARDCODED_PLY_RE.search(line)
            if match:
                violations.append(f"{rel_path}:{idx}: hardcoded asset path '{match.group(0)}'")

    if violations:
        print("[benchmark-asset-guard] hardcoded benchmark/synthetic asset paths are not allowed")
        for violation in violations:
            print(f"  - {violation}")
        print("[benchmark-asset-guard] use benchmark_asset_manifest.json + benchmark_scene_contract.gd resolution")
        return 1

    print("[benchmark-asset-guard] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

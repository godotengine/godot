#!/usr/bin/env python3
"""Validate Gaussian Splatting shader dependency wiring in SCons scripts.

This guard protects ISSUE-034 by asserting the build graph keeps generated
shader headers connected to their GLSL sources/includes and to module objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = REPO_ROOT / "modules" / "gaussian_splatting"
ROOT_SCSUB = MODULE_ROOT / "SCsub"
COMPUTE_SCSUB = MODULE_ROOT / "compute" / "SCsub"
SHADERS_SCSUB = MODULE_ROOT / "shaders" / "SCsub"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed reading '{path}': {exc}") from exc


def _require_contains(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{context} is missing required contract fragment: {needle}")


def main() -> int:
    failures: list[str] = []

    root_scsub = _read(ROOT_SCSUB)
    compute_scsub = _read(COMPUTE_SCSUB)
    shaders_scsub = _read(SHADERS_SCSUB)

    # Root build graph must consume generated headers from both shader domains.
    _require_contains(root_scsub, 'compute_generated_headers = SConscript("compute/SCsub")',
            ROOT_SCSUB.name, failures)
    _require_contains(root_scsub, 'shader_generated_headers = SConscript("shaders/SCsub")',
            ROOT_SCSUB.name, failures)
    _require_contains(root_scsub, "Depends(module_sources, generated_shader_headers)",
            ROOT_SCSUB.name, failures)

    # Compute shaders must depend on shared include files and builder implementation.
    _require_contains(compute_scsub, 'Glob("../shaders/includes/*.glsl")',
            COMPUTE_SCSUB.as_posix(), failures)
    _require_contains(compute_scsub, '#glsl_builders.py',
            COMPUTE_SCSUB.as_posix(), failures)
    _require_contains(compute_scsub, "env.Depends(",
            COMPUTE_SCSUB.as_posix(), failures)

    # Tile/painterly shader generation must also pin include + builder dependencies.
    _require_contains(shaders_scsub, 'gl_include_files = [str(f) for f in Glob("includes/*.glsl")]',
            SHADERS_SCSUB.as_posix(), failures)
    _require_contains(shaders_scsub, '#glsl_builders.py',
            SHADERS_SCSUB.as_posix(), failures)
    _require_contains(shaders_scsub, "env.Depends([f + \".gen.h\" for f in glsl_files]",
            SHADERS_SCSUB.as_posix(), failures)

    if failures:
        print("[shader-dependency-check] FAILED")
        for failure in failures:
            print(f"[shader-dependency-check] - {failure}")
        return 1

    print("[shader-dependency-check] PASSED")
    print("[shader-dependency-check] Shader include/build graph contracts are intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate deterministic synthetic PLY fixtures used by CI/runtime flows.

Policy:
- Only canonical synthetic PLY paths are allowed.
- Legacy large/hand-managed PLY names are not generated and are treated as forbidden.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class PLYSpec:
    relative_path: str
    count: int
    seed: int
    radius: float


CANONICAL_SPECS: tuple[PLYSpec, ...] = (
    PLYSpec("tests/fixtures/test_splats.ply", 1024, 1101, 3.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/test_splats.ply", 1024, 1101, 3.0),
    PLYSpec("templates/gaussian_splat_template/assets/template_splats.ply", 768, 2202, 2.4),
)

FORBIDDEN_LEGACY_PLYS: tuple[str, ...] = (
    "tests/examples/godot/test_project/cabin.ply",
    "tests/examples/godot/test_project/ancient-corinth-clean.ply",
    "tests/examples/godot/test_project/splat_12000.ply",
    "tests/examples/godot/test_project/scenes/5x5%23-5_-10_0_-5%23-1_-2.ply",
    "tests/examples/godot/test_project/test_splats.ply",
)


def _header(count: int) -> bytes:
    text = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )
    return text.encode("ascii")


def _encode_splat(
    x: float,
    y: float,
    z: float,
    color_rgb: tuple[float, float, float],
    scale: float,
    opacity: float,
) -> tuple[float, ...]:
    opacity = max(0.01, min(0.99, opacity))
    scale = max(0.02, scale)

    opacity_logit = math.log(opacity / (1.0 - opacity))
    scale_log = math.log(scale)
    r, g, b = (channel / SH_C0 for channel in color_rgb)

    return (
        float(x),
        float(y),
        float(z),
        float(r),
        float(g),
        float(b),
        float(opacity_logit),
        float(scale_log),
        float(scale_log),
        float(scale_log),
        1.0,
        0.0,
        0.0,
        0.0,
    )


def _generate_rows(spec: PLYSpec) -> list[tuple[float, ...]]:
    rng = random.Random(spec.seed)
    rows: list[tuple[float, ...]] = []

    palette: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.1, 0.1),
        (0.1, 1.0, 0.1),
        (0.1, 0.1, 1.0),
        (1.0, 0.8, 0.2),
        (0.9, 0.3, 0.9),
    )

    for index in range(spec.count):
        theta = rng.random() * (2.0 * math.pi)
        phi = math.acos(2.0 * rng.random() - 1.0)
        radius = spec.radius * pow(rng.random(), 0.33)

        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = -4.0 + radius * math.cos(phi)

        color = palette[index % len(palette)]
        scale = 0.08 + 0.35 * rng.random()
        opacity = 0.55 + 0.4 * rng.random()
        rows.append(_encode_splat(x, y, z, color, scale, opacity))

    return rows


def _write_ply(path: Path, rows: list[tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_header(len(rows)))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


def _resolve_repo_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _check_only(repo_root: Path) -> int:
    missing: list[str] = []
    forbidden_present: list[str] = []

    for spec in CANONICAL_SPECS:
        file_path = repo_root / spec.relative_path
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            missing.append(spec.relative_path)

    for rel_path in FORBIDDEN_LEGACY_PLYS:
        file_path = repo_root / rel_path
        if file_path.is_file() and file_path.stat().st_size > 0:
            forbidden_present.append(rel_path)

    if missing or forbidden_present:
        print("[prepare_synthetic_assets] synthetic asset policy check failed")
        if missing:
            print("  missing canonical assets:")
            for rel in missing:
                print(f"    - {rel}")
        if forbidden_present:
            print("  forbidden legacy assets present:")
            for rel in forbidden_present:
                print(f"    - {rel}")
        return 1

    print("[prepare_synthetic_assets] synthetic asset policy check passed")
    return 0


def _generate(repo_root: Path, quiet: bool) -> int:
    removed: list[str] = []

    for spec in CANONICAL_SPECS:
        output = repo_root / spec.relative_path
        rows = _generate_rows(spec)
        _write_ply(output, rows)
        if not quiet:
            print(
                f"[prepare_synthetic_assets] wrote {spec.count:5d} splats -> {spec.relative_path}"
            )

    for rel_path in FORBIDDEN_LEGACY_PLYS:
        file_path = repo_root / rel_path
        if file_path.exists():
            file_path.unlink()
            removed.append(rel_path)

    if not quiet and removed:
        print("[prepare_synthetic_assets] removed forbidden legacy assets:")
        for rel in removed:
            print(f"  - {rel}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic canonical synthetic PLY fixtures."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (defaults to auto-detected root from script location).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify canonical synthetic assets and forbidden-asset policy.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output during generation.",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root(args.repo_root)
    if not repo_root.is_dir():
        print(f"[prepare_synthetic_assets] invalid repo root: {repo_root}")
        return 1

    if args.check:
        return _check_only(repo_root)
    return _generate(repo_root, args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())

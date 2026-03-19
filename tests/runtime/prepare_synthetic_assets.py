#!/usr/bin/env python3
"""Generate deterministic synthetic fixtures used by benchmark/runtime flows.

Policy:
- Only canonical generated fixture paths are allowed.
- Legacy hand-managed fixture assets are forbidden.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class PLYSpec:
    relative_path: str
    count: int
    seed: int
    pattern: str
    scale: float

CANONICAL_SPECS: tuple[PLYSpec, ...] = (
    PLYSpec("tests/fixtures/test_splats.ply", 1024, 1101, "sphere", 3.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/test_splats.ply", 1024, 1101, "sphere", 3.0),
    PLYSpec("templates/gaussian_splat_template/assets/template_splats.ply", 768, 2202, "sphere", 2.4),
    PLYSpec("tests/fixtures/synthetic_sphere.ply", 2048, 3101, "sphere", 4.5),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_sphere.ply", 2048, 3101, "sphere", 4.5),
    PLYSpec("tests/fixtures/synthetic_cube.ply", 2048, 3201, "cube", 7.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_cube.ply", 2048, 3201, "cube", 7.0),
    PLYSpec("tests/fixtures/synthetic_plane.ply", 2048, 3301, "plane", 9.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_plane.ply", 2048, 3301, "plane", 9.0),
    PLYSpec("tests/fixtures/synthetic_torus.ply", 3072, 3401, "torus", 6.5),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_torus.ply", 3072, 3401, "torus", 6.5),
    PLYSpec("tests/fixtures/synthetic_spiral.ply", 3072, 3501, "spiral", 8.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_spiral.ply", 3072, 3501, "spiral", 8.0),
    PLYSpec("tests/fixtures/synthetic_mandelbulb.ply", 4096, 3601, "mandelbulb", 1.4),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_mandelbulb.ply", 4096, 3601, "mandelbulb", 1.4),
    PLYSpec("tests/fixtures/synthetic_cloud.ply", 4096, 3701, "cloud", 12.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_cloud.ply", 4096, 3701, "cloud", 12.0),
    PLYSpec("tests/fixtures/synthetic_flower_field.ply", 4096, 3801, "flower_field", 10.0),
    PLYSpec("tests/examples/godot/test_project/tests/fixtures/synthetic_flower_field.ply", 4096, 3801, "flower_field", 10.0),
)

FORBIDDEN_LEGACY_PLYS: tuple[str, ...] = (
    "tests/examples/godot/test_project/cabin.ply",
    "tests/examples/godot/test_project/ancient-corinth-clean.ply",
    "tests/examples/godot/test_project/splat_12000.ply",
    "tests/examples/godot/test_project/scenes/5x5%23-5_-10_0_-5%23-1_-2.ply",
    "tests/examples/godot/test_project/test_splats.ply",
)

FORBIDDEN_LEGACY_ASSET_DIRS: tuple[str, ...] = (
    "tests/examples/godot/test_project/benchmark_assets_generated",
)

FORBIDDEN_LEGACY_PATH_TOKENS: tuple[str, ...] = (
    "benchmark_assets_generated/",
)

CANONICAL_MANIFESTS: tuple[str, ...] = (
    "tests/fixtures/benchmark_asset_manifest.json",
    "tests/examples/godot/test_project/tests/fixtures/benchmark_asset_manifest.json",
)

SCENE_DEFAULT_ASSETS: dict[str, str] = {
    "benchmark_suite_lane": "res://tests/fixtures/test_splats.ply",
    "benchmark_unified": "res://tests/fixtures/test_splats.ply",
    "benchmark_small_baseline": "res://tests/fixtures/test_splats.ply",
    "synthetic_sphere": "res://tests/fixtures/synthetic_sphere.ply",
    "synthetic_cube": "res://tests/fixtures/synthetic_cube.ply",
    "synthetic_plane": "res://tests/fixtures/synthetic_plane.ply",
    "synthetic_torus": "res://tests/fixtures/synthetic_torus.ply",
    "synthetic_spiral": "res://tests/fixtures/synthetic_spiral.ply",
    "synthetic_mandelbulb": "res://tests/fixtures/synthetic_mandelbulb.ply",
    "synthetic_cloud": "res://tests/fixtures/synthetic_cloud.ply",
    "synthetic_flower_field": "res://tests/fixtures/synthetic_flower_field.ply",
}

LANE_DEFAULT_ASSETS: dict[str, str] = {
    "static_baseline": "res://tests/fixtures/test_splats.ply",
    "streaming_corridor": "res://tests/fixtures/test_splats.ply",
    "city_flyover": "res://tests/fixtures/test_splats.ply",
    "instance_storm": "res://tests/fixtures/test_splats.ply",
    "lighting_stress": "res://tests/fixtures/test_splats.ply",
    "animation_arena": "res://tests/fixtures/test_splats.ply",
    "lod_torture": "res://tests/fixtures/test_splats.ply",
    "integrity_sentinel": "res://tests/fixtures/test_splats.ply",
    "parity_fidelity": "res://tests/fixtures/test_splats.ply",
    "long_soak": "res://tests/fixtures/test_splats.ply",
    "unified_composite": "res://tests/fixtures/test_splats.ply",
    "small_baseline": "res://tests/fixtures/test_splats.ply",
    "instance_pipeline_ab": "res://tests/fixtures/test_splats.ply",
    "synthetic_sphere": "res://tests/fixtures/synthetic_sphere.ply",
    "synthetic_cube": "res://tests/fixtures/synthetic_cube.ply",
    "synthetic_plane": "res://tests/fixtures/synthetic_plane.ply",
    "synthetic_torus": "res://tests/fixtures/synthetic_torus.ply",
    "synthetic_spiral": "res://tests/fixtures/synthetic_spiral.ply",
    "synthetic_mandelbulb": "res://tests/fixtures/synthetic_mandelbulb.ply",
    "synthetic_cloud": "res://tests/fixtures/synthetic_cloud.ply",
    "synthetic_flower_field": "res://tests/fixtures/synthetic_flower_field.ply",
}


def _benchmark_asset_manifest() -> dict[str, object]:
    return {
        "version": "2.0.0",
        "default_asset": "res://tests/fixtures/test_splats.ply",
        "scene_defaults": dict(SCENE_DEFAULT_ASSETS),
        "lane_defaults": dict(LANE_DEFAULT_ASSETS),
    }


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

    def _sample_sphere(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        theta = rng.random() * (2.0 * math.pi)
        phi = math.acos(2.0 * rng.random() - 1.0)
        radius = spec.scale * pow(rng.random(), 0.33)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = -4.0 + radius * math.cos(phi)
        color = palette[index % len(palette)]
        scale = 0.08 + 0.35 * rng.random()
        opacity = 0.55 + 0.4 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_cube(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        half = spec.scale * 0.5
        x = rng.uniform(-half, half)
        y = rng.uniform(-half, half)
        z = -4.0 + rng.uniform(-half, half)
        color = palette[index % len(palette)]
        scale = 0.09 + 0.24 * rng.random()
        opacity = 0.6 + 0.3 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_plane(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        x = rng.uniform(-spec.scale, spec.scale)
        y = rng.uniform(-0.2, 0.2)
        z = -4.0 + rng.uniform(-spec.scale, spec.scale)
        u = (x / (spec.scale * 2.0)) + 0.5
        v = (z + 4.0 + spec.scale) / (spec.scale * 2.0)
        color = (
            0.2 + 0.7 * max(0.0, min(1.0, u)),
            0.35 + 0.55 * max(0.0, min(1.0, v)),
            0.45 + 0.4 * rng.random(),
        )
        scale = 0.06 + 0.2 * rng.random()
        opacity = 0.55 + 0.35 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_torus(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        major = spec.scale
        minor = max(0.8, spec.scale * 0.32)
        u = rng.uniform(0.0, 2.0 * math.pi)
        v = rng.uniform(0.0, 2.0 * math.pi)
        ring = major + minor * math.cos(v)
        x = ring * math.cos(u)
        y = minor * math.sin(v)
        z = -4.0 + ring * math.sin(u)
        color = (
            0.5 + 0.45 * math.sin(u),
            0.45 + 0.4 * math.sin(v + 1.2),
            0.5 + 0.45 * math.cos(u + v),
        )
        scale = 0.05 + 0.18 * rng.random()
        opacity = 0.62 + 0.28 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_spiral(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        t = float(index) / max(1.0, float(spec.count - 1))
        angle = t * 14.0 * math.pi
        radius = 1.0 + spec.scale * (0.25 + 0.75 * t)
        x = radius * math.cos(angle) + rng.uniform(-0.22, 0.22)
        y = (t - 0.5) * spec.scale * 1.8 + rng.uniform(-0.1, 0.1)
        z = -4.0 + radius * math.sin(angle) + rng.uniform(-0.22, 0.22)
        color = (
            0.35 + 0.55 * math.sin(angle * 0.2 + 0.4),
            0.35 + 0.55 * math.sin(angle * 0.23 + 2.0),
            0.35 + 0.55 * math.sin(angle * 0.27 + 4.0),
        )
        scale = 0.05 + 0.16 * rng.random()
        opacity = 0.6 + 0.32 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_cloud(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        ex = spec.scale
        ey = spec.scale * 0.3
        ez = spec.scale * 0.7
        while True:
            x = rng.uniform(-ex, ex)
            y = rng.uniform(-ey, ey)
            z = rng.uniform(-ez, ez)
            d = (x / ex) ** 2 + (y / ey) ** 2 + (z / ez) ** 2
            if d <= 1.0:
                break
        z_world = -4.0 + z
        height = max(0.0, min(1.0, (y + ey) / (2.0 * ey)))
        shade = max(0.0, min(1.0, 1.0 - d))
        color = (
            0.65 + 0.3 * height,
            0.68 + 0.28 * height,
            0.75 + 0.2 * height + 0.05 * shade,
        )
        scale = 0.1 + 0.38 * shade
        opacity = 0.22 + 0.55 * shade
        return x, y + 4.0, z_world, color, scale, opacity

    def _sample_flower_field(index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        petals = 6
        flower_idx = index // petals
        petal_idx = index % petals
        rng_flower = random.Random(spec.seed + flower_idx * 17)
        cx = rng_flower.uniform(-spec.scale, spec.scale)
        cz = rng_flower.uniform(-spec.scale, spec.scale) - 4.0
        stem_h = rng_flower.uniform(0.4, 1.4)
        angle = (petal_idx / float(petals)) * 2.0 * math.pi + rng_flower.uniform(-0.2, 0.2)
        petal_r = rng_flower.uniform(0.25, 0.6)
        x = cx + math.cos(angle) * petal_r
        y = stem_h + rng.uniform(-0.04, 0.05)
        z = cz + math.sin(angle) * petal_r
        if petal_idx == 0:
            color = (0.95, 0.85, 0.12)
            scale = 0.07 + 0.07 * rng.random()
        else:
            color = (
                0.5 + 0.45 * math.sin(angle * 1.1 + 0.2),
                0.4 + 0.5 * math.sin(angle * 1.3 + 2.3),
                0.4 + 0.5 * math.sin(angle * 1.7 + 4.4),
            )
            scale = 0.08 + 0.14 * rng.random()
        opacity = 0.72 + 0.23 * rng.random()
        return x, y, z, color, scale, opacity

    def _sample_mandelbulb(_index: int) -> tuple[float, float, float, tuple[float, float, float], float, float]:
        attempts = 0
        while attempts < 60:
            attempts += 1
            x = rng.uniform(-spec.scale, spec.scale)
            y = rng.uniform(-spec.scale, spec.scale)
            z = rng.uniform(-spec.scale, spec.scale)
            zx, zy, zz = x, y, z
            escaped = False
            for _ in range(8):
                r = math.sqrt(zx * zx + zy * zy + zz * zz)
                if r > 2.0:
                    escaped = True
                    break
                if r < 1e-6:
                    theta = 0.0
                    phi = 0.0
                else:
                    theta = math.acos(zz / r)
                    phi = math.atan2(zy, zx)
                power = 8.0
                rp = r ** power
                theta *= power
                phi *= power
                sin_t = math.sin(theta)
                zx = rp * sin_t * math.cos(phi) + x
                zy = rp * sin_t * math.sin(phi) + y
                zz = rp * math.cos(theta) + z
            if escaped:
                intensity = max(0.0, min(1.0, math.sqrt(x * x + y * y + z * z) / (spec.scale * 1.2)))
                color = (
                    0.45 + 0.5 * math.sin(8.0 * intensity + 0.2),
                    0.45 + 0.5 * math.sin(8.0 * intensity + 2.2),
                    0.45 + 0.5 * math.sin(8.0 * intensity + 4.2),
                )
                scale = 0.06 + 0.16 * (1.0 - intensity)
                opacity = 0.55 + 0.38 * (1.0 - intensity)
                return x * 5.0, y * 5.0, z * 5.0 - 4.0, color, scale, opacity
        return _sample_spiral(rng.randint(0, spec.count - 1))

    samplers = {
        "sphere": _sample_sphere,
        "cube": _sample_cube,
        "plane": _sample_plane,
        "torus": _sample_torus,
        "spiral": _sample_spiral,
        "cloud": _sample_cloud,
        "flower_field": _sample_flower_field,
        "mandelbulb": _sample_mandelbulb,
    }

    if spec.pattern not in samplers:
        raise ValueError(f"Unsupported pattern '{spec.pattern}' for {spec.relative_path}")

    sampler = samplers[spec.pattern]
    for index in range(spec.count):
        x, y, z, color, scale, opacity = sampler(index)
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


def _write_manifest(repo_root: Path) -> None:
    manifest = _benchmark_asset_manifest()
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    for rel_path in CANONICAL_MANIFESTS:
        path = repo_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def _benchmark_scene_script_paths(repo_root: Path) -> list[str]:
    root = repo_root / "tests" / "examples" / "godot" / "test_project" / "scenes"
    if not root.is_dir():
        return []
    out: list[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".gd", ".tscn"}:
            continue
        out.append(str(file_path.relative_to(repo_root)).replace("\\", "/"))
    return sorted(out)


def _check_only(repo_root: Path) -> int:
    missing: list[str] = []
    forbidden_present: list[str] = []
    legacy_path_references: list[str] = []

    for spec in CANONICAL_SPECS:
        file_path = repo_root / spec.relative_path
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            missing.append(spec.relative_path)

    for rel_path in CANONICAL_MANIFESTS:
        file_path = repo_root / rel_path
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            missing.append(rel_path)

    for rel_path in FORBIDDEN_LEGACY_PLYS:
        file_path = repo_root / rel_path
        if file_path.is_file() and file_path.stat().st_size > 0:
            forbidden_present.append(rel_path)

    for rel_dir in FORBIDDEN_LEGACY_ASSET_DIRS:
        dir_path = repo_root / rel_dir
        if not dir_path.is_dir():
            continue
        for ply_file in sorted(dir_path.rglob("*.ply")):
            forbidden_present.append(str(ply_file.relative_to(repo_root)).replace("\\", "/"))

    for rel_path in _benchmark_scene_script_paths(repo_root):
        file_path = repo_root / rel_path
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in FORBIDDEN_LEGACY_PATH_TOKENS:
            if token in text:
                legacy_path_references.append(f"{rel_path}: contains legacy token '{token}'")
                break

    if missing or forbidden_present or legacy_path_references:
        print("[prepare_synthetic_assets] synthetic asset policy check failed")
        if missing:
            print("  missing canonical assets:")
            for rel in missing:
                print(f"    - {rel}")
        if forbidden_present:
            print("  forbidden legacy assets present:")
            for rel in forbidden_present:
                print(f"    - {rel}")
        if legacy_path_references:
            print("  forbidden legacy path references:")
            for rel in legacy_path_references:
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
                f"[prepare_synthetic_assets] wrote {spec.count:5d} splats ({spec.pattern}) -> {spec.relative_path}"
            )

    _write_manifest(repo_root)
    if not quiet:
        for rel_path in CANONICAL_MANIFESTS:
            print(f"[prepare_synthetic_assets] wrote manifest -> {rel_path}")

    for rel_path in FORBIDDEN_LEGACY_PLYS:
        file_path = repo_root / rel_path
        if file_path.exists():
            file_path.unlink()
            removed.append(rel_path)

    for rel_dir in FORBIDDEN_LEGACY_ASSET_DIRS:
        dir_path = repo_root / rel_dir
        if dir_path.is_dir():
            shutil.rmtree(dir_path)
            removed.append(rel_dir + "/")

    if not quiet and removed:
        print("[prepare_synthetic_assets] removed forbidden legacy assets:")
        for rel in removed:
            print(f"  - {rel}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic canonical benchmark/synthetic fixtures."
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

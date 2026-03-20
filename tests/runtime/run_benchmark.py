#!/usr/bin/env python3
"""
Run the GodotGS benchmark lane suite with profile-aware durations and aggregation.

This runner executes benchmark scenes one-by-one, collects each lane JSON report,
and writes a suite-level report + markdown summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LaneDefinition:
    lane_id: str
    scene: str
    description: str
    durations: dict[str, float]
    weights: dict[str, float]


LANES: list[LaneDefinition] = [
    LaneDefinition(
        lane_id="static_baseline",
        scene="res://scenes/benchmark_suite/lane_static_baseline.tscn",
        description="Low-noise raster baseline",
        durations={"quick": 10.0, "performance": 20.0, "showcase": 25.0},
        weights={"quick": 12.0, "performance": 10.0, "showcase": 8.0},
    ),
    LaneDefinition(
        lane_id="streaming_corridor",
        scene="res://scenes/benchmark_suite/lane_streaming_corridor.tscn",
        description="Camera sweep stressing chunk turnover",
        durations={"quick": 12.0, "performance": 40.0, "showcase": 50.0},
        weights={"quick": 12.0, "performance": 12.0, "showcase": 12.0},
    ),
    LaneDefinition(
        lane_id="city_flyover",
        scene="res://scenes/benchmark_suite/lane_city_flyover.tscn",
        description="High-altitude visibility-change stress",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 50.0},
        weights={"quick": 10.0, "performance": 10.0, "showcase": 12.0},
    ),
    LaneDefinition(
        lane_id="instance_storm",
        scene="res://scenes/benchmark_suite/lane_instance_storm.tscn",
        description="Many-instance submission pressure",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 45.0},
        weights={"quick": 10.0, "performance": 10.0, "showcase": 10.0},
    ),
    LaneDefinition(
        lane_id="lighting_stress",
        scene="res://scenes/benchmark_suite/lane_lighting_stress.tscn",
        description="Animated light and shading stress",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 50.0},
        weights={"quick": 10.0, "performance": 10.0, "showcase": 12.0},
    ),
    LaneDefinition(
        lane_id="animation_arena",
        scene="res://scenes/benchmark_suite/lane_animation_arena.tscn",
        description="Wind + effectors + transform updates",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 45.0},
        weights={"quick": 10.0, "performance": 10.0, "showcase": 10.0},
    ),
    LaneDefinition(
        lane_id="lod_torture",
        scene="res://scenes/benchmark_suite/lane_lod_torture.tscn",
        description="Near/far pulsing LOD churn",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 40.0},
        weights={"quick": 10.0, "performance": 12.0, "showcase": 10.0},
    ),
    LaneDefinition(
        lane_id="integrity_sentinel",
        scene="res://scenes/benchmark_suite/lane_integrity_sentinel.tscn",
        description="Deterministic view sweeps for artifact detection",
        durations={"quick": 10.0, "performance": 25.0, "showcase": 35.0},
        weights={"quick": 8.0, "performance": 8.0, "showcase": 8.0},
    ),
    LaneDefinition(
        lane_id="long_soak",
        scene="res://scenes/benchmark_suite/lane_long_soak.tscn",
        description="Long-horizon drift/churn stability",
        durations={"quick": 20.0, "performance": 120.0, "showcase": 150.0},
        weights={"quick": 3.0, "performance": 8.0, "showcase": 8.0},
    ),
    LaneDefinition(
        lane_id="unified_composite",
        scene="res://scenes/benchmark_unified.tscn",
        description="Integrated all-systems composite lane",
        durations={"quick": 45.0, "performance": 180.0, "showcase": 240.0},
        weights={"quick": 15.0, "performance": 20.0, "showcase": 10.0},
    ),
    LaneDefinition(
        lane_id="parity_fidelity",
        scene="res://scenes/benchmark_suite/lane_parity_fidelity.tscn",
        description="Parity-fidelity locked lane (minimal quality throttling)",
        durations={"quick": 12.0, "performance": 35.0, "showcase": 45.0, "parity": 35.0},
        weights={"quick": 8.0, "performance": 12.0, "showcase": 12.0, "parity": 20.0},
    ),
    LaneDefinition(
        lane_id="small_baseline",
        scene="res://scenes/benchmark_small_baseline.tscn",
        description="Small baseline benchmark lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 20.0},
        weights={"quick": 5.0, "performance": 5.0, "everything": 5.0},
    ),
    LaneDefinition(
        lane_id="synthetic_sphere",
        scene="res://scenes/synthetic_sphere.tscn",
        description="Synthetic sphere lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 18.0, "synthetic-only": 15.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 8.0},
    ),
    LaneDefinition(
        lane_id="synthetic_cube",
        scene="res://scenes/synthetic_cube.tscn",
        description="Synthetic cube lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 18.0, "synthetic-only": 15.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 8.0},
    ),
    LaneDefinition(
        lane_id="synthetic_plane",
        scene="res://scenes/synthetic_plane.tscn",
        description="Synthetic plane lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 18.0, "synthetic-only": 15.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 8.0},
    ),
    LaneDefinition(
        lane_id="synthetic_torus",
        scene="res://scenes/synthetic_torus.tscn",
        description="Synthetic torus lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 18.0, "synthetic-only": 15.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 8.0},
    ),
    LaneDefinition(
        lane_id="synthetic_spiral",
        scene="res://scenes/synthetic_spiral.tscn",
        description="Synthetic spiral lane",
        durations={"quick": 12.0, "performance": 20.0, "everything": 18.0, "synthetic-only": 15.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 8.0},
    ),
    LaneDefinition(
        lane_id="synthetic_mandelbulb",
        scene="res://scenes/synthetic_mandelbulb.tscn",
        description="Synthetic mandelbulb lane",
        durations={"quick": 15.0, "performance": 30.0, "everything": 25.0, "synthetic-only": 25.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 10.0},
    ),
    LaneDefinition(
        lane_id="synthetic_cloud",
        scene="res://scenes/synthetic_cloud.tscn",
        description="Synthetic cloud lane",
        durations={"quick": 15.0, "performance": 30.0, "everything": 25.0, "synthetic-only": 25.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 10.0},
    ),
    LaneDefinition(
        lane_id="synthetic_flower_field",
        scene="res://scenes/synthetic_flower_field.tscn",
        description="Synthetic flower-field lane",
        durations={"quick": 15.0, "performance": 30.0, "everything": 25.0, "synthetic-only": 25.0},
        weights={"quick": 2.0, "performance": 2.0, "everything": 2.0, "synthetic-only": 10.0},
    ),
    LaneDefinition(
        lane_id="instance_pipeline_ab_serial",
        scene="res://scenes/benchmark_suite/lane_instance_pipeline_ab.tscn",
        description="A/B serial lane",
        durations={"quick": 20.0, "performance": 35.0, "everything": 35.0, "ab-only": 35.0},
        weights={"quick": 4.0, "performance": 6.0, "everything": 6.0, "ab-only": 10.0},
    ),
    LaneDefinition(
        lane_id="instance_pipeline_ab_single_pass",
        scene="res://scenes/benchmark_suite/lane_instance_pipeline_ab.tscn",
        description="A/B single-pass lane",
        durations={"quick": 20.0, "performance": 35.0, "everything": 35.0, "ab-only": 35.0},
        weights={"quick": 4.0, "performance": 6.0, "everything": 6.0, "ab-only": 10.0},
    ),
]

LANE_INDEX_BY_ID = {lane.lane_id: idx for idx, lane in enumerate(LANES)}
PROFILE_DEFAULT_LANE_IDS: dict[str, tuple[str, ...]] = {
    "everything": tuple(
        lane_id
        for lane_id in (
            "static_baseline",
            "streaming_corridor",
            "city_flyover",
            "instance_storm",
            "lighting_stress",
            "animation_arena",
            "lod_torture",
            "integrity_sentinel",
            "long_soak",
            "parity_fidelity",
            "unified_composite",
            "small_baseline",
            "synthetic_sphere",
            "synthetic_cube",
            "synthetic_plane",
            "synthetic_torus",
            "synthetic_spiral",
            "synthetic_mandelbulb",
            "synthetic_cloud",
            "synthetic_flower_field",
        )
    ),
    "quick": (
        "static_baseline",
        "streaming_corridor",
        "integrity_sentinel",
        "unified_composite",
        "small_baseline",
        "synthetic_sphere",
    ),
    "performance": (
        "static_baseline",
        "streaming_corridor",
        "city_flyover",
        "instance_storm",
        "lighting_stress",
        "animation_arena",
        "lod_torture",
        "integrity_sentinel",
        "long_soak",
        "parity_fidelity",
        "unified_composite",
        "small_baseline",
    ),
    "synthetic-only": (
        "synthetic_sphere",
        "synthetic_cube",
        "synthetic_plane",
        "synthetic_torus",
        "synthetic_spiral",
        "synthetic_mandelbulb",
        "synthetic_cloud",
        "synthetic_flower_field",
    ),
    "ab-only": (
        "instance_pipeline_ab_serial",
        "instance_pipeline_ab_single_pass",
    ),
}
LANE_TIMEOUT_MIN_SECONDS = 90
LANE_TIMEOUT_GRACE_SECONDS = 45
LANE_TIMEOUT_SCALE = 3.0
PERF_FORBIDDEN_SORT_ROUTE_TOKENS: tuple[str, ...] = ("CPU_FALLBACK",)
PROFILE_WARMUP_SECONDS: dict[str, float] = {
    "quick": 2.0,
    "performance": 5.0,
    "everything": 5.0,
    "synthetic-only": 2.0,
    "ab-only": 5.0,
    "showcase": 8.0,
    "parity": 5.0,
}
DEFAULT_CAPTURE_LANES: tuple[str, ...] = ("integrity_sentinel", "parity_fidelity")
TEXT_DEPENDENCY_SUFFIXES: tuple[str, ...] = (".tscn", ".gd", ".tres", ".tscn")
RES_PATH_PATTERN = re.compile(r"res://[^\s\"'\]\)]+")
ORCHESTRATOR_SCENE = "res://scenes/benchmark_orchestrator.tscn"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_godot_binary(repo_root: Path) -> Path:
    legacy_path = repo_root / "bin" / "godot.linuxbsd.editor.dev.x86_64"
    split_path = repo_root / "godot-source" / "bin" / "godot.linuxbsd.editor.dev.x86_64"
    if legacy_path.exists():
        return legacy_path
    return split_path


def _default_project_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "examples" / "godot" / "test_project"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _lane_timeout_seconds(
    lane_duration: float,
    *,
    timeout_scale: float | None = None,
    timeout_grace: int | None = None,
) -> int:
    scale = timeout_scale if timeout_scale is not None else LANE_TIMEOUT_SCALE
    grace = timeout_grace if timeout_grace is not None else LANE_TIMEOUT_GRACE_SECONDS
    scaled = int(math.ceil(lane_duration * scale))
    return max(LANE_TIMEOUT_MIN_SECONDS, scaled + grace)


def _lane_warmup_seconds(profile: str, lane_duration: float) -> float:
    baseline = float(PROFILE_WARMUP_SECONDS.get(profile, 3.0))
    return max(0.0, min(baseline, max(0.0, lane_duration - 1.0)))


def _coerce_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _report_renderer_metric(report: dict[str, Any], key: str, default: Any = None) -> Any:
    telemetry = report.get("renderer_telemetry")
    if isinstance(telemetry, dict) and key in telemetry:
        return telemetry.get(key, default)
    overall = report.get("overall")
    if isinstance(overall, dict):
        return overall.get(key, default)
    return default


def _normalize_execution_mode_token(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("single_pass"):
        return "single_pass"
    if raw.startswith("serial"):
        return "serial"
    if raw == "auto":
        return "auto"
    return ""


def _lane_requires_gpu_timestamps(args: argparse.Namespace, lane_id: str) -> bool:
    if bool(args.require_gpu_timestamps):
        return True
    if args.profile == "parity":
        return True
    return lane_id == "parity_fidelity"


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Run GodotGS benchmark lane suite.")
    parser.add_argument(
        "--godot-binary",
        default=os.environ.get("GODOT_BINARY", str(_default_godot_binary(repo_root))),
        help="Path to Godot executable.",
    )
    parser.add_argument(
        "--project-path",
        default=str(_default_project_path(repo_root)),
        help="Godot test project path.",
    )
    parser.add_argument(
        "--profile",
        choices=["everything", "quick", "performance", "synthetic-only", "ab-only"],
        default="everything",
        help="Suite profile controls lane durations and aggregate weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "tests" / "output" / "benchmark_suite" / _now_stamp()),
        help="Directory for lane JSON/log outputs and suite report.",
    )
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        help="Run only selected lane IDs. Can be repeated.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first lane failure.",
    )
    parser.add_argument("--capture", action="store_true", help="Enable capture output.")
    parser.add_argument(
        "--capture-dir",
        default="",
        help="Optional directory for benchmark screenshot captures. Defaults to <output-dir>/captures when captures are enabled.",
    )
    parser.add_argument(
        "--reference-dir",
        default="",
        help="Optional directory containing reference PNGs for SSIM/PSNR comparisons.",
    )
    parser.add_argument(
        "--capture-lane",
        action="append",
        default=[],
        help="Lane ID to capture screenshots for. Can be repeated. Defaults to deterministic lanes when captures are enabled.",
    )
    parser.add_argument(
        "--no-captures",
        action="store_true",
        help="Disable screenshot capture and visual comparisons.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable benchmark dashboard/chart generation.",
    )
    return parser.parse_args()


def _ensure_duration_scale(value: float) -> float:
    if value <= 0.0:
        raise ValueError("--duration-scale must be > 0")
    return value


def _select_lanes(requested: list[str]) -> list[LaneDefinition]:
    if not requested:
        return LANES
    index = {lane.lane_id: lane for lane in LANES}
    selected: list[LaneDefinition] = []
    missing: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for lane_id in requested:
        if lane_id in seen:
            duplicates.append(lane_id)
            continue
        seen.add(lane_id)
        lane = index.get(lane_id)
        if lane is None:
            missing.append(lane_id)
            continue
        selected.append(lane)
    if duplicates:
        raise ValueError(f"Duplicate lane ids: {', '.join(duplicates)}")
    if missing:
        raise ValueError(f"Unknown lane ids: {', '.join(missing)}")
    return selected


def _select_capture_lanes(
    selected_lanes: list[LaneDefinition],
    requested: list[str],
    *,
    captures_disabled: bool,
    references_requested: bool,
) -> set[str]:
    if captures_disabled:
        return set()
    valid_lane_ids = {lane.lane_id for lane in selected_lanes}
    if requested:
        requested_set = set(requested)
        missing = sorted(requested_set - valid_lane_ids)
        if missing:
            raise ValueError(f"Unknown capture lane ids: {', '.join(missing)}")
        return requested_set
    # Default to deterministic capture lanes whenever captures are enabled,
    # regardless of whether --reference-dir was supplied.
    return {lane_id for lane_id in DEFAULT_CAPTURE_LANES if lane_id in valid_lane_ids}


def _load_asset_manifest(path: str) -> dict[str, str]:
    if not path:
        return {}
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"--asset-manifest file not found: {manifest_path}") from exc
    except OSError as exc:
        raise ValueError(f"--asset-manifest could not be read: {manifest_path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"--asset-manifest must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--asset-manifest must be a JSON object")
    out: dict[str, str] = {}
    malformed_entries: list[str] = []
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            malformed_entries.append(repr(key))
            continue
        out[key] = value
    if malformed_entries:
        raise ValueError(
            "--asset-manifest entries must map string lane IDs to string asset paths; "
            f"invalid keys: {', '.join(malformed_entries)}"
        )
    return out


def _resolve_res_path(project_path: Path, resource_path: str) -> Path:
    return project_path / resource_path[len("res://") :]


def _extract_res_dependencies(text: str) -> set[str]:
    return set(match.group(0) for match in RES_PATH_PATTERN.finditer(text))


def _collect_missing_res_dependencies(
    project_path: Path,
    root_resource: str,
    *,
    source_label: str,
) -> list[str]:
    to_visit: list[tuple[str, str]] = [(root_resource, source_label)]
    visited: set[str] = set()
    missing: list[str] = []

    while to_visit:
        resource_path, owner = to_visit.pop()
        if resource_path in visited:
            continue
        visited.add(resource_path)

        if not resource_path.startswith("res://"):
            continue

        resource_file = _resolve_res_path(project_path, resource_path)
        if not resource_file.exists():
            missing.append(f"{resource_path} (referenced from {owner})")
            continue

        if resource_file.suffix not in TEXT_DEPENDENCY_SUFFIXES:
            continue

        try:
            text = resource_file.read_text(encoding="utf-8")
        except OSError as exc:
            missing.append(f"{resource_path} (could not read: {exc})")
            continue

        for child in sorted(_extract_res_dependencies(text)):
            to_visit.append((child, resource_path))

    return missing


def _validate_suite_dependencies(
    project_path: Path,
    lanes: list[LaneDefinition],
    asset_manifest: dict[str, str],
    generated_assets: dict[str, str],
) -> list[str]:
    failures: list[str] = []
    for lane in lanes:
        scene_failures = _collect_missing_res_dependencies(
            project_path,
            lane.scene,
            source_label=f"lane:{lane.lane_id}",
        )
        failures.extend(f"lane={lane.lane_id}: {entry}" for entry in scene_failures)

        asset_override = asset_manifest.get(lane.lane_id, generated_assets.get(lane.lane_id, ""))
        if not asset_override:
            continue
        if asset_override.startswith("res://"):
            asset_path = _resolve_res_path(project_path, asset_override)
        else:
            asset_path = Path(asset_override)
        if not asset_path.exists():
            failures.append(
                f"lane={lane.lane_id}: asset override missing: {asset_override}"
            )
    return failures


def _write_dummy_gaussian_ply(path: Path, vertex_count: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
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
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        for idx in range(vertex_count):
            # Multi-ring, multi-layer structure for complex geometry
            layer = idx % 12
            ring_base = 1.0 + (layer * 0.8)
            ring = ring_base + rng.uniform(-0.3, 0.3)

            # Spiral pattern with noise
            angle = (2.0 * math.pi * float(idx) / max(1.0, float(vertex_count))) * 8.0 + rng.uniform(-0.15, 0.15)
            height = -5.0 + (float(idx) / max(1.0, float(vertex_count))) * 10.0 + rng.uniform(-0.5, 0.5)

            x = math.cos(angle) * ring + rng.uniform(-0.2, 0.2)
            y = height
            z = math.sin(angle) * ring + rng.uniform(-0.2, 0.2)

            # Varied colors based on position
            c0 = 0.3 + 0.5 * math.sin(angle * 0.5 + layer * 0.3)
            c1 = 0.3 + 0.5 * math.cos(angle * 0.7 + layer * 0.2)
            c2 = 0.4 + 0.4 * math.sin(height * 0.2 + 0.7)

            opacity = 0.85 + rng.uniform(-0.1, 0.15)
            # Larger splats for visibility (log scale, so -2.5 is bigger than -3.0)
            scale = -2.2 + rng.uniform(-0.3, 0.3)

            rot_w = 1.0
            rot_x = rng.uniform(-0.1, 0.1)
            rot_y = rng.uniform(-0.1, 0.1)
            rot_z = rng.uniform(-0.1, 0.1)
            packed = struct.pack(
                "<14f",
                float(x),
                float(y),
                float(z),
                float(c0),
                float(c1),
                float(c2),
                float(opacity),
                float(scale),
                float(scale),
                float(scale),
                float(rot_w),
                float(rot_x),
                float(rot_y),
                float(rot_z),
            )
            fh.write(packed)


def _remove_legacy_gsplatworld_cache_artifacts(ply_path: Path) -> None:
    legacy_world = ply_path.with_suffix(".gsplatworld")
    legacy_world_import = Path(str(legacy_world) + ".import")
    for artifact in (legacy_world, legacy_world_import):
        try:
            artifact.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Best-effort cleanup only; benchmark generation should still proceed.
            pass


def _generate_dummy_assets(project_path: Path, lanes: list[LaneDefinition], output_dir_name: str) -> dict[str, str]:
    out_dir = project_path / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    # Splat counts per lane for meaningful benchmarks
    LANE_SPLAT_COUNTS = {
        "static_baseline": 100_000,
        "streaming_corridor": 250_000,
        "instance_storm": 500_000,
        "lod_torture": 1_000_000,
        "lighting_stress": 150_000,
        "animation_arena": 200_000,
        "city_flyover": 500_000,
        "long_soak": 250_000,
        "integrity_sentinel": 100_000,
    }
    DEFAULT_SPLAT_COUNT = 100_000
    for lane in lanes:
        lane_index = LANE_INDEX_BY_ID.get(lane.lane_id)
        if lane_index is None:
            lane_index = sum((i + 1) * ord(ch) for i, ch in enumerate(lane.lane_id))
        vertex_count = LANE_SPLAT_COUNTS.get(lane.lane_id, DEFAULT_SPLAT_COUNT)
        out_path = out_dir / f"{lane.lane_id}.ply"
        _remove_legacy_gsplatworld_cache_artifacts(out_path)
        _write_dummy_gaussian_ply(out_path, vertex_count=vertex_count, seed=9100 + lane_index)
        mapping[lane.lane_id] = f"res://{output_dir_name}/{lane.lane_id}.ply"
    return mapping


def _run_lane(
    godot_binary: Path,
    project_path: Path,
    profile: str,
    lane: LaneDefinition,
    duration_scale: float,
    output_dir: Path,
    asset_override: str,
    pass_headless_summary: bool,
    benchmark_instancing_mode: str,
    capture_dir: Path | None,
    reference_dir: Path | None,
    timeout_scale: float | None = None,
    timeout_grace: int | None = None,
) -> dict[str, Any]:
    base_duration = lane.durations.get(profile, lane.durations.get("performance", 20.0))
    lane_duration = max(5.0, base_duration * duration_scale)
    lane_warmup = _lane_warmup_seconds(profile, lane_duration)
    lane_timeout_seconds = _lane_timeout_seconds(
        lane_duration, timeout_scale=timeout_scale, timeout_grace=timeout_grace,
    )
    lane_json = output_dir / f"{lane.lane_id}.json"
    lane_log = output_dir / f"{lane.lane_id}.log"
    if lane_json.exists():
        lane_json.unlink()

    cmd = [
        str(godot_binary),
        "--disable-vsync",
        "--path",
        str(project_path),
        "--scene",
        lane.scene,
        "--benchmark-duration",
        f"{lane_duration:.3f}",
        "--benchmark-warmup",
        f"{lane_warmup:.3f}",
        "--benchmark-output",
        str(lane_json),
        "--benchmark-lane-tag",
        profile,
    ]
    if pass_headless_summary:
        cmd.append("--benchmark-headless-summary")
    if asset_override:
        cmd.append(f"--benchmark-asset={asset_override}")
    if benchmark_instancing_mode != "auto":
        cmd.append(f"--benchmark-instancing-mode={benchmark_instancing_mode}")
    if capture_dir is not None:
        cmd.extend(["--benchmark-capture-dir", str(capture_dir)])
        cmd.extend(["--benchmark-capture-tag", profile])
    if reference_dir is not None:
        cmd.extend(["--benchmark-reference-dir", str(reference_dir)])

    timed_out = False
    timeout_message = ""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=lane_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_message = (
            f"Lane timed out after {lane_timeout_seconds}s "
            f"(expected_duration={lane_duration:.1f}s)."
        )
        timeout_stdout = _coerce_subprocess_text(exc.stdout)
        timeout_stderr = _coerce_subprocess_text(exc.stderr)
        if timeout_message:
            timeout_stderr = f"{timeout_stderr}\n{timeout_message}" if timeout_stderr else timeout_message
        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=timeout_stdout,
            stderr=timeout_stderr,
        )

    with lane_log.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.write(f"# lane_timeout_seconds={lane_timeout_seconds}\n")
        if timed_out:
            fh.write(f"# TIMEOUT: {timeout_message}\n")
        if completed.stdout:
            fh.write("STDOUT:\n")
            fh.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                fh.write("\n")
        if completed.stderr:
            fh.write("\nSTDERR:\n")
            fh.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                fh.write("\n")

    report: dict[str, Any] | None = None
    if lane_json.exists():
        try:
            report = json.loads(lane_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None

    score = None
    avg_fps = None
    p1_fps = None
    p99_frame_ms = None
    sample_count = None
    summary_source = "overall"
    warmup_p1_fps = None
    steady_p1_fps = None
    warmup_sample_count = None
    steady_sample_count = None
    sort_sync_fallback_count = None
    sort_total_route_fallback_count = None
    sort_cached_fallback_count = None
    sort_identity_fallback_count = None
    sort_cull_order_fallback_count = None
    gpu_timeline_stall_count = None
    gpu_timeline_stall_ms = None
    streaming_queue_pressure_active = None
    streaming_queue_pressure_frames = None
    streaming_vram_cap_hit_frames = None
    sort_active_algorithm = None
    stage_sort_status = None
    stage_sort_reason = None
    instancing_execution_mode = None
    instancing_execution_path = None
    instancing_execution_reason = None
    instance_pipeline_true_single_pass_enabled = None
    stage_cull_time_ms = None
    stage_sort_time_ms = None
    stage_raster_time_ms = None
    overlap_records = None
    overlap_record_budget = None
    overlap_record_budget_effective = None
    overlap_thinning_keep_ratio = None
    effective_quality_preset = None
    effective_max_splats = None
    effective_lod_enabled = None
    effective_distance_cull_enabled = None
    effective_tiny_splat_screen_radius = None
    gpu_timing_available = None
    gpu_frame_time_source = None
    gpu_frame_estimate_ms = None
    gpu_time_frame_ms = None
    node_visible_splats_max = None
    node_total_visible_splats_max = None
    node_primary_visible_splats_max = None
    primary_node_quality = None
    capture_count = None
    capture_saved_count = None
    capture_reference_match_count = None
    capture_threshold_pass_count = None
    capture_ssim_min = None
    capture_ssim_avg = None
    capture_psnr_min = None
    capture_psnr_avg = None
    if isinstance(report, dict):
        score = report.get("score")
        summary = report.get("overall", {})
        steady = report.get("steady_overall", {})
        warmup = report.get("warmup_overall", {})
        if isinstance(steady, dict) and int(steady.get("sample_count", 0)) > 0:
            summary = steady
            summary_source = "steady_overall"
            steady_score = report.get("steady_score")
            if isinstance(steady_score, (int, float)):
                score = steady_score
        if isinstance(summary, dict):
            avg_fps = summary.get("avg_fps")
            p1_fps = summary.get("p1_fps")
            p99_frame_ms = summary.get("p99_frame_ms")
            sample_count = summary.get("sample_count")
        if isinstance(warmup, dict):
            warmup_p1_fps = warmup.get("p1_fps")
            warmup_sample_count = warmup.get("sample_count")
        if isinstance(steady, dict):
            steady_p1_fps = steady.get("p1_fps")
            steady_sample_count = steady.get("sample_count")
        sort_sync_fallback_count = _report_renderer_metric(report, "sort_sync_fallback_count")
        sort_total_route_fallback_count = _report_renderer_metric(report, "sort_total_route_fallback_count")
        sort_cached_fallback_count = _report_renderer_metric(report, "sort_cached_fallback_count")
        sort_identity_fallback_count = _report_renderer_metric(report, "sort_identity_fallback_count")
        sort_cull_order_fallback_count = _report_renderer_metric(report, "sort_cull_order_fallback_count")
        gpu_timeline_stall_count = _report_renderer_metric(report, "gpu_timeline_stall_count")
        gpu_timeline_stall_ms = _report_renderer_metric(report, "gpu_timeline_stall_ms")
        streaming_queue_pressure_active = _report_renderer_metric(report, "streaming_queue_pressure_active")
        streaming_queue_pressure_frames = _report_renderer_metric(report, "streaming_queue_pressure_frames")
        streaming_vram_cap_hit_frames = _report_renderer_metric(report, "streaming_vram_cap_hit_frames")
        sort_active_algorithm = _report_renderer_metric(report, "sort_active_algorithm")
        stage_sort_status = _report_renderer_metric(report, "stage_sort_status")
        stage_sort_reason = _report_renderer_metric(report, "stage_sort_reason")
        instancing_execution_mode = _report_renderer_metric(report, "instance_pipeline_execution_mode")
        instancing_execution_path = _report_renderer_metric(report, "instance_pipeline_execution_path")
        instancing_execution_reason = _report_renderer_metric(report, "instance_pipeline_execution_reason")
        instance_pipeline_true_single_pass_enabled = _report_renderer_metric(
            report, "instance_pipeline_true_single_pass_enabled"
        )
        stage_cull_time_ms = _report_renderer_metric(report, "stage_cull_time_ms")
        stage_sort_time_ms = _report_renderer_metric(report, "stage_sort_time_ms")
        stage_raster_time_ms = _report_renderer_metric(report, "stage_raster_time_ms")
        overlap_records = _report_renderer_metric(report, "overlap_records")
        overlap_record_budget = _report_renderer_metric(report, "overlap_record_budget")
        overlap_record_budget_effective = _report_renderer_metric(report, "overlap_record_budget_effective")
        overlap_thinning_keep_ratio = _report_renderer_metric(report, "overlap_thinning_keep_ratio")
        effective_quality_preset = _report_renderer_metric(report, "effective_quality_preset")
        effective_max_splats = _report_renderer_metric(report, "effective_max_splats")
        effective_lod_enabled = _report_renderer_metric(report, "effective_lod_enabled")
        effective_distance_cull_enabled = _report_renderer_metric(report, "effective_distance_cull_enabled")
        effective_tiny_splat_screen_radius = _report_renderer_metric(report, "effective_tiny_splat_screen_radius")
        gpu_timing_available = _report_renderer_metric(report, "gpu_timing_available")
        gpu_frame_time_source = _report_renderer_metric(report, "gpu_frame_time_source")
        gpu_frame_estimate_ms = _report_renderer_metric(report, "gpu_frame_estimate_ms")
        gpu_time_frame_ms = _report_renderer_metric(report, "gpu_frame_time_ms")
        node_visible_splats_max = report.get("node_visible_splats_max")
        node_total_visible_splats_max = report.get("node_total_visible_splats_max")
        node_primary_visible_splats_max = report.get("node_primary_visible_splats_max")
        primary_node_quality = report.get("primary_node_quality")
        visual_summary = report.get("visual_summary")
        if isinstance(visual_summary, dict):
            capture_count = visual_summary.get("capture_count")
            capture_saved_count = visual_summary.get("saved_capture_count")
            capture_reference_match_count = visual_summary.get("reference_match_count")
            capture_threshold_pass_count = visual_summary.get("threshold_pass_count")
            capture_ssim_min = visual_summary.get("ssim_min")
            capture_ssim_avg = visual_summary.get("ssim_avg")
            capture_psnr_min = visual_summary.get("psnr_min")
            capture_psnr_avg = visual_summary.get("psnr_avg")

    return {
        "lane_id": lane.lane_id,
        "lane_name": lane.description,
        "scene": lane.scene,
        "profile_duration_s": base_duration,
        "applied_duration_s": lane_duration,
        "warmup_duration_s": lane_warmup,
        "weight": lane.weights.get(profile, lane.weights.get("performance", 10.0)),
        "asset_override": asset_override,
        "instancing_mode": benchmark_instancing_mode,
        "exit_code": completed.returncode,
        "timed_out": timed_out,
        "lane_timeout_seconds": lane_timeout_seconds,
        "json_path": str(lane_json),
        "log_path": str(lane_log),
        "score": score,
        "avg_fps": avg_fps,
        "p1_fps": p1_fps,
        "p99_frame_ms": p99_frame_ms,
        "sample_count": sample_count,
        "warmup_p1_fps": warmup_p1_fps,
        "steady_p1_fps": steady_p1_fps,
        "warmup_sample_count": warmup_sample_count,
        "steady_sample_count": steady_sample_count,
        "sort_sync_fallback_count": sort_sync_fallback_count,
        "sort_total_route_fallback_count": sort_total_route_fallback_count,
        "sort_cached_fallback_count": sort_cached_fallback_count,
        "sort_identity_fallback_count": sort_identity_fallback_count,
        "sort_cull_order_fallback_count": sort_cull_order_fallback_count,
        "gpu_timeline_stall_count": gpu_timeline_stall_count,
        "gpu_timeline_stall_ms": gpu_timeline_stall_ms,
        "streaming_queue_pressure_active": streaming_queue_pressure_active,
        "streaming_queue_pressure_frames": streaming_queue_pressure_frames,
        "streaming_vram_cap_hit_frames": streaming_vram_cap_hit_frames,
        "sort_active_algorithm": sort_active_algorithm,
        "stage_sort_status": stage_sort_status,
        "stage_sort_reason": stage_sort_reason,
        "instancing_execution_mode": instancing_execution_mode,
        "instancing_execution_path": instancing_execution_path,
        "instancing_execution_reason": instancing_execution_reason,
        "instance_pipeline_true_single_pass_enabled": instance_pipeline_true_single_pass_enabled,
        "stage_cull_time_ms": stage_cull_time_ms,
        "stage_sort_time_ms": stage_sort_time_ms,
        "stage_raster_time_ms": stage_raster_time_ms,
        "overlap_records": overlap_records,
        "overlap_record_budget": overlap_record_budget,
        "overlap_record_budget_effective": overlap_record_budget_effective,
        "overlap_thinning_keep_ratio": overlap_thinning_keep_ratio,
        "instancing_execution_mode_normalized": _normalize_execution_mode_token(instancing_execution_mode),
        "effective_quality_preset": effective_quality_preset,
        "effective_max_splats": effective_max_splats,
        "effective_lod_enabled": effective_lod_enabled,
        "effective_distance_cull_enabled": effective_distance_cull_enabled,
        "effective_tiny_splat_screen_radius": effective_tiny_splat_screen_radius,
        "gpu_timing_available": gpu_timing_available,
        "gpu_frame_time_source": gpu_frame_time_source,
        "gpu_frame_estimate_ms": gpu_frame_estimate_ms,
        "gpu_time_frame_ms": gpu_time_frame_ms,
        "node_visible_splats_max": node_visible_splats_max,
        "node_total_visible_splats_max": node_total_visible_splats_max,
        "node_primary_visible_splats_max": node_primary_visible_splats_max,
        "primary_node_quality": primary_node_quality,
        "capture_count": capture_count,
        "capture_saved_count": capture_saved_count,
        "capture_reference_match_count": capture_reference_match_count,
        "capture_threshold_pass_count": capture_threshold_pass_count,
        "capture_ssim_min": capture_ssim_min,
        "capture_ssim_avg": capture_ssim_avg,
        "capture_psnr_min": capture_psnr_min,
        "capture_psnr_avg": capture_psnr_avg,
        "capture_dir": str(capture_dir) if capture_dir is not None else "",
        "reference_dir": str(reference_dir) if reference_dir is not None else "",
        "summary_source": summary_source,
        "report": report,
    }


def _metric_as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        metric = float(value)
        if not math.isfinite(metric):
            return None
        return metric
    return None


def _lane_has_visible_output(report: dict[str, Any]) -> tuple[bool, str]:
    metric_samples: list[tuple[str, float, int]] = []

    node_visible = _metric_as_float(report.get("node_visible_splats_max"))
    if node_visible is not None:
        metric_samples.append(("node_visible_splats_max", node_visible, 0))
    node_total_visible = _metric_as_float(report.get("node_total_visible_splats_max"))
    if node_total_visible is not None:
        metric_samples.append(("node_total_visible_splats_max", node_total_visible, 0))

    monitor_max = report.get("monitor_max")
    if isinstance(monitor_max, dict):
        monitor_visible = _metric_as_float(monitor_max.get("visible_splats"))
        if monitor_visible is not None:
            metric_samples.append(("monitor_max.visible_splats", monitor_visible, 1))

    if not metric_samples:
        return False, "missing visibility metrics"
    for _, value, _ in metric_samples:
        if value > 0.0:
            return True, ""

    detail_parts: list[str] = []
    for metric_name, value, decimals in metric_samples:
        detail_parts.append(f"{metric_name}={value:.{decimals}f}")
    return False, ", ".join(detail_parts)


def _lane_uses_forbidden_cpu_sort_route(report: dict[str, Any]) -> tuple[bool, str]:
    overall = report.get("overall")
    if not isinstance(overall, dict):
        return False, ""
    sort_route_uid = overall.get("sort_route_uid")
    if not isinstance(sort_route_uid, str):
        return False, ""
    normalized = sort_route_uid.strip().upper()
    if not normalized:
        return False, ""
    for token in PERF_FORBIDDEN_SORT_ROUTE_TOKENS:
        if token in normalized:
            return True, sort_route_uid
    return False, ""


def _lane_supports_asset_override(lane: LaneDefinition) -> bool:
    return lane.lane_id != "unified_composite"


def _compute_aggregate(profile: str, lane_results: list[dict[str, Any]]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for result in lane_results:
        if not bool(result.get("lane_valid", True)):
            continue
        score = result.get("score")
        weight = float(result.get("weight", 0.0))
        if isinstance(score, (int, float)):
            weighted_sum += float(score) * weight
            weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return weighted_sum / weight_sum


def _write_suite_summary_markdown(
    report_path: Path,
    profile: str,
    aggregate_score: float,
    lane_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# Benchmark Suite Summary",
        "",
        f"- Profile: `{profile}`",
        f"- Aggregate score: `{aggregate_score:.2f}`",
        "",
        "| Lane | Source | Score | Avg FPS | P1 FPS | Steady P1 | Warmup P1 | Sync FB | Route FB | Sort ms | Raster ms | Visible | GPU ms | GPU Src | Eff Preset | Eff Max | DistCull | Stall Cnt | Q Pressure | Exec Mode | Mode OK | P99 ms | Samples | Captures | SSIM min | PSNR min | Exit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in lane_results:
        def _fmt(v: Any, decimals: int = 2) -> str:
            if isinstance(v, (int, float)):
                return f"{float(v):.{decimals}f}"
            return "n/a"

        lines.append(
            "| "
            f"`{lane['lane_id']}` | "
            f"`{lane.get('summary_source', 'overall')}` | "
            f"{_fmt(lane.get('score'))} | "
            f"{_fmt(lane.get('avg_fps'))} | "
            f"{_fmt(lane.get('p1_fps'))} | "
            f"{_fmt(lane.get('steady_p1_fps'))} | "
            f"{_fmt(lane.get('warmup_p1_fps'))} | "
            f"{_fmt(lane.get('sort_sync_fallback_count'), 0)} | "
            f"{_fmt(lane.get('sort_total_route_fallback_count'), 0)} | "
            f"{_fmt(lane.get('stage_sort_time_ms'))} | "
            f"{_fmt(lane.get('stage_raster_time_ms'))} | "
            f"{_fmt(lane.get('node_total_visible_splats_max') if lane.get('node_total_visible_splats_max') is not None else lane.get('node_visible_splats_max'), 0)} | "
            f"{_fmt(lane.get('gpu_time_frame_ms'))} | "
            f"{lane.get('gpu_frame_time_source', 'n/a')} | "
            f"{lane.get('effective_quality_preset', 'n/a')} | "
            f"{_fmt(lane.get('effective_max_splats'), 0)} | "
            f"{lane.get('effective_distance_cull_enabled', 'n/a')} | "
            f"{_fmt(lane.get('gpu_timeline_stall_count'), 0)} | "
            f"{_fmt(lane.get('streaming_queue_pressure_frames'), 0)} | "
            f"{lane.get('instancing_execution_mode', 'n/a')} | "
            f"{'yes' if lane.get('instancing_mode_match', True) else 'no'} | "
            f"{_fmt(lane.get('p99_frame_ms'))} | "
            f"{lane.get('sample_count', 'n/a')} | "
            f"{_fmt(lane.get('capture_count'), 0)} | "
            f"{_fmt(lane.get('capture_ssim_min'), 3)} | "
            f"{_fmt(lane.get('capture_psnr_min'))} | "
            f"{lane.get('exit_code', 'n/a')} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_dashboard(suite_report_path: Path, output_dir: Path) -> list[str]:
    dashboard_script = _repo_root() / "scripts" / "generate_benchmark_suite_dashboard.py"
    if not dashboard_script.exists():
        print(f"[suite] dashboard script missing: {dashboard_script}", file=sys.stderr)
        return []
    completed = subprocess.run(
        [sys.executable, str(dashboard_script), "--suite-report", str(suite_report_path), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
        print(f"[suite] dashboard generation failed with exit={completed.returncode}", file=sys.stderr)
        return []
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        if line.startswith("[dashboard] wrote "):
            paths.append(line.replace("[dashboard] wrote ", "", 1).strip())
    return paths


def main() -> int:
    args = _parse_args()
    try:
        duration_scale = _ensure_duration_scale(args.duration_scale)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    godot_binary = Path(args.godot_binary).resolve()
    project_path = Path(args.project_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not godot_binary.exists():
        print(f"ERROR: Godot binary not found: {godot_binary}", file=sys.stderr)
        return 2
    if not project_path.exists():
        print(f"ERROR: Project path not found: {project_path}", file=sys.stderr)
        return 2

    try:
        selected_lanes = _select_lanes(args.lane)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        capture_lane_ids = _select_capture_lanes(
            selected_lanes,
            args.capture_lane,
            captures_disabled=bool(args.no_captures),
            references_requested=bool(args.reference_dir),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        asset_manifest = _load_asset_manifest(args.asset_manifest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    generated_assets: dict[str, str] = {}
    if args.generate_dummy_assets:
        generated_assets = _generate_dummy_assets(project_path, selected_lanes, args.dummy_asset_dir)

    preflight_failures = _validate_suite_dependencies(
        project_path=project_path,
        lanes=selected_lanes,
        asset_manifest=asset_manifest,
        generated_assets=generated_assets,
    )
    if preflight_failures:
        print("ERROR: benchmark-suite preflight failed due to missing res:// dependencies:", file=sys.stderr)
        for failure in preflight_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    lane_results: list[dict[str, Any]] = []
    failed = False
    capture_dir = None if not capture_lane_ids else Path(args.capture_dir).resolve() if args.capture_dir else (output_dir / "captures")
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = None
    if args.reference_dir and capture_lane_ids:
        reference_dir = Path(args.reference_dir).resolve()
        if not reference_dir.exists():
            print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
            return 2
    if args.reference_dir and not capture_lane_ids:
        print(
            "ERROR: --reference-dir was provided but no capture lanes are active. "
            "Specify --capture-lane or remove --reference-dir.",
            file=sys.stderr,
        )
        return 2
    for lane in selected_lanes:
        asset_override = asset_manifest.get(lane.lane_id, generated_assets.get(lane.lane_id, ""))
        lane_base_duration = lane.durations.get(args.profile, lane.durations.get("performance", 20.0))
        lane_duration = max(5.0, lane_base_duration * duration_scale)
        if asset_override and not _lane_supports_asset_override(lane):
            print(
                f"[suite] lane={lane.lane_id} ignores --benchmark-asset; skipping asset override."
            )
            asset_override = ""
        print(f"[suite] running lane={lane.lane_id} scene={lane.scene} duration={lane_duration:.1f}s")
        result = _run_lane(
            godot_binary=godot_binary,
            project_path=project_path,
            profile=args.profile,
            lane=lane,
            duration_scale=duration_scale,
            output_dir=output_dir,
            asset_override=asset_override,
            pass_headless_summary=not args.no_headless_summary,
            benchmark_instancing_mode=args.benchmark_instancing_mode,
            capture_dir=capture_dir if lane.lane_id in capture_lane_ids else None,
            reference_dir=reference_dir if lane.lane_id in capture_lane_ids else None,
            timeout_scale=args.timeout_scale,
            timeout_grace=args.timeout_grace,
        )
        lane_results.append(result)
        score = result.get("score")
        avg_fps = result.get("avg_fps")
        print(
            f"[suite] lane={lane.lane_id} exit={result['exit_code']} "
            f"score={score if score is not None else 'n/a'} avg_fps={avg_fps if avg_fps is not None else 'n/a'} "
            f"source={result.get('summary_source', 'overall')} sync_fb={result.get('sort_sync_fallback_count', 'n/a')} "
            f"route_fb={result.get('sort_total_route_fallback_count', 'n/a')} "
            f"sort_ms={result.get('stage_sort_time_ms', 'n/a')} raster_ms={result.get('stage_raster_time_ms', 'n/a')} "
            f"visible={result.get('node_total_visible_splats_max', result.get('node_visible_splats_max', 'n/a'))} "
            f"gpu_ms={result.get('gpu_time_frame_ms', 'n/a')} "
            f"gpu_src={result.get('gpu_frame_time_source', 'n/a')} "
            f"effective_preset={result.get('effective_quality_preset', 'n/a')} "
            f"effective_max_splats={result.get('effective_max_splats', 'n/a')} "
            f"distance_cull={result.get('effective_distance_cull_enabled', 'n/a')} "
            f"stall={result.get('gpu_timeline_stall_count', 'n/a')} "
            f"q_pressure={result.get('streaming_queue_pressure_frames', 'n/a')} "
            f"exec_mode={result.get('instancing_execution_mode', 'n/a')} "
            f"exec_path={result.get('instancing_execution_path', 'n/a')} "
            f"exec_reason={result.get('instancing_execution_reason', 'n/a')}"
        )
        report_valid = isinstance(result.get("report"), dict)
        visible_output_valid = False
        visibility_failure_detail = ""
        cpu_sort_route_detected = False
        cpu_sort_route_uid = ""
        instancing_mode_enforced = args.benchmark_instancing_mode in {"serial", "single_pass"}
        instancing_mode_match = True
        require_gpu_timestamps = _lane_requires_gpu_timestamps(args, lane.lane_id)
        gpu_timing_match = True
        visual_reference_enforced = bool(result.get("reference_dir")) and int(result.get("capture_count") or 0) > 0
        visual_reference_match = True
        enforce_no_cpu_sort_route = args.profile == "performance"
        if report_valid:
            visible_output_valid, visibility_failure_detail = _lane_has_visible_output(result["report"])
            if enforce_no_cpu_sort_route:
                cpu_sort_route_detected, cpu_sort_route_uid = _lane_uses_forbidden_cpu_sort_route(result["report"])
            if instancing_mode_enforced:
                instance_count = int(result["report"].get("instance_count", 0))
                if instance_count > 1:
                    actual_mode = _normalize_execution_mode_token(result.get("instancing_execution_mode", ""))
                    instancing_mode_match = actual_mode == args.benchmark_instancing_mode
            if require_gpu_timestamps:
                gpu_timing_match = bool(result.get("gpu_timing_available"))
            if visual_reference_enforced:
                capture_count_for_reference = int(result.get("capture_count") or 0)
                matched = int(result.get("capture_reference_match_count") or 0)
                passed = int(result.get("capture_threshold_pass_count") or 0)
                visual_reference_match = (
                    capture_count_for_reference > 0
                    and matched == capture_count_for_reference
                    and passed == capture_count_for_reference
                )
        lane_valid = (
            int(result["exit_code"]) == 0
            and report_valid
            and visible_output_valid
            and not cpu_sort_route_detected
            and instancing_mode_match
            and gpu_timing_match
            and visual_reference_match
        )
        result["report_valid"] = report_valid
        result["visible_output_valid"] = visible_output_valid
        result["cpu_sort_route_enforced"] = enforce_no_cpu_sort_route
        result["cpu_sort_route_detected"] = cpu_sort_route_detected
        result["cpu_sort_route_uid"] = cpu_sort_route_uid
        result["instancing_mode_enforced"] = instancing_mode_enforced
        result["instancing_mode_match"] = instancing_mode_match
        result["gpu_timing_required"] = require_gpu_timestamps
        result["gpu_timing_match"] = gpu_timing_match
        result["visual_reference_enforced"] = visual_reference_enforced
        result["visual_reference_match"] = visual_reference_match
        result["lane_valid"] = lane_valid
        lane_failed = not lane_valid
        if lane_failed:
            if bool(result.get("timed_out")):
                print(
                    f"[suite] lane={lane.lane_id} timed out after {result.get('lane_timeout_seconds')}s",
                    file=sys.stderr,
                )
            elif not report_valid:
                print(
                    f"[suite] lane={lane.lane_id} missing/invalid JSON report: {result['json_path']}",
                    file=sys.stderr,
                )
            elif int(result["exit_code"]) == 0 and not visible_output_valid:
                print(
                    f"[suite] lane={lane.lane_id} invalid report: expected visible splats > 0 ({visibility_failure_detail})",
                    file=sys.stderr,
                )
            elif cpu_sort_route_detected:
                print(
                    f"[suite] lane={lane.lane_id} invalid report: CPU sort route detected during performance profile ({cpu_sort_route_uid})",
                    file=sys.stderr,
                )
            elif not instancing_mode_match:
                print(
                    f"[suite] lane={lane.lane_id} invalid report: requested instancing mode "
                    f"{args.benchmark_instancing_mode} but executed {result.get('instancing_execution_mode', 'unknown')}",
                    file=sys.stderr,
                )
            elif not gpu_timing_match:
                print(
                    f"[suite] lane={lane.lane_id} invalid report: GPU timestamps unavailable "
                    f"(source={result.get('gpu_frame_time_source', 'unknown')})",
                    file=sys.stderr,
                )
            elif not visual_reference_match:
                print(
                    f"[suite] lane={lane.lane_id} invalid report: visual references missing or below threshold "
                    f"(matched={result.get('capture_reference_match_count', 0)} passed={result.get('capture_threshold_pass_count', 0)})",
                    file=sys.stderr,
                )
            elif int(result["exit_code"]) != 0:
                print(
                    f"[suite] lane={lane.lane_id} exited with code {result['exit_code']}: {result['log_path']}",
                    file=sys.stderr,
                )
            failed = True
            if args.fail_fast:
                break

    aggregate_score = _compute_aggregate(args.profile, lane_results)
    require_gpu_timestamps_global = bool(
        args.require_gpu_timestamps
        or args.profile == "parity"
        or any(bool(lane_result.get("gpu_timing_required")) for lane_result in lane_results)
    )

    suite_report = {
        "name": "GodotGS Benchmark Suite",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "profile": args.profile,
        "duration_scale": duration_scale,
        "godot_binary": str(godot_binary),
        "project_path": str(project_path),
        "output_dir": str(output_dir),
        "generate_dummy_assets": bool(args.generate_dummy_assets),
        "dummy_asset_dir": args.dummy_asset_dir,
        "asset_manifest_path": args.asset_manifest,
        "benchmark_instancing_mode": args.benchmark_instancing_mode,
        "capture_lane_ids": sorted(capture_lane_ids),
        "capture_dir": str(capture_dir) if capture_dir is not None else "",
        "reference_dir": str(reference_dir) if reference_dir is not None else "",
        "require_gpu_timestamps": require_gpu_timestamps_global,
        "require_gpu_timestamps_lanes": [
            lane_result.get("lane_id")
            for lane_result in lane_results
            if bool(lane_result.get("gpu_timing_required"))
        ],
        "aggregate_score": aggregate_score,
        "lane_results": lane_results,
    }

    suite_report_path = output_dir / "benchmark_suite_report.json"
    suite_report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")

    summary_md_path = output_dir / "benchmark_suite_summary.md"
    _write_suite_summary_markdown(summary_md_path, args.profile, aggregate_score, lane_results)

    dashboard_paths: list[str] = []
    if not args.no_dashboard:
        dashboard_paths = _generate_dashboard(suite_report_path, output_dir)
        if dashboard_paths:
            suite_report["dashboard_artifacts"] = dashboard_paths
            suite_report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")

    print(f"[suite] aggregate_score={aggregate_score:.2f}")
    print(f"[suite] report={suite_report_path}")
    print(f"[suite] summary={summary_md_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

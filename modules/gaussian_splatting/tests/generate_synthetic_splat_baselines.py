#!/usr/bin/env python3
"""
Generate and validate deterministic synthetic splat baseline artifacts.

This mirrors the deterministic sampling and summary hashing used in
`synthetic_splat_generators.cpp` so baseline artifacts stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

U64_MASK = (1 << 64) - 1
HASH_BASIS = 1469598103934665603
HASH_PRIME = 1099511628211
HASH_QUANTIZATION_SCALE = 100000.0
CMP_EPSILON2 = 1e-12

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "synthetic_baselines"

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Quat4 = Tuple[float, float, float, float]
Color4 = Tuple[float, float, float, float]


def _u64(value: int) -> int:
    return value & U64_MASK


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _hex_u64(value: int) -> str:
    return f"0x{_u64(value):016x}"


def _fnv1a_u64(hash_value: int, value: int) -> int:
    return _u64((hash_value ^ _u64(value)) * HASH_PRIME)


def _hash_bool(hash_value: int, value: bool) -> int:
    return _fnv1a_u64(hash_value, 1 if value else 0)


def _hash_u64(hash_value: int, value: int) -> int:
    return _fnv1a_u64(hash_value, value)


def _hash_i64(hash_value: int, value: int) -> int:
    return _fnv1a_u64(hash_value, value)


def _hash_float(hash_value: int, value: float) -> int:
    quantized = _round_half_away_from_zero(_f32(value) * HASH_QUANTIZATION_SCALE)
    return _hash_i64(hash_value, quantized)


def _hash_vector3(hash_value: int, vec: Vec3) -> int:
    hash_value = _hash_float(hash_value, vec[0])
    hash_value = _hash_float(hash_value, vec[1])
    hash_value = _hash_float(hash_value, vec[2])
    return hash_value


def _hash_vector2(hash_value: int, vec: Vec2) -> int:
    hash_value = _hash_float(hash_value, vec[0])
    hash_value = _hash_float(hash_value, vec[1])
    return hash_value


def _hash_quaternion(hash_value: int, quat: Quat4) -> int:
    hash_value = _hash_float(hash_value, quat[0])
    hash_value = _hash_float(hash_value, quat[1])
    hash_value = _hash_float(hash_value, quat[2])
    hash_value = _hash_float(hash_value, quat[3])
    return hash_value


def _hash_color(hash_value: int, color: Color4) -> int:
    hash_value = _hash_float(hash_value, color[0])
    hash_value = _hash_float(hash_value, color[1])
    hash_value = _hash_float(hash_value, color[2])
    hash_value = _hash_float(hash_value, color[3])
    return hash_value


class DeterministicRng:
    def __init__(self, seed: int):
        self.state = _u64(seed)

    def next_u64(self) -> int:
        self.state = _u64(self.state + 0x9E3779B97F4A7C15)
        z = self.state
        z = _u64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = _u64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return _u64(z ^ (z >> 31))

    def next_unit_float(self) -> float:
        bits = self.next_u64() >> 40
        return _f32(bits * (1.0 / float(1 << 24)))

    def range(self, minimum: float, maximum: float) -> float:
        minimum_f = _f32(minimum)
        maximum_f = _f32(maximum)
        return _f32(minimum_f + _f32(maximum_f - minimum_f) * self.next_unit_float())


def _normalize_range(minimum: float, maximum: float) -> Tuple[float, float]:
    if maximum < minimum:
        return maximum, minimum
    return minimum, maximum


def _normalize_range_vec3(minimum: Vec3, maximum: Vec3) -> Tuple[Vec3, Vec3]:
    min_list = [minimum[0], minimum[1], minimum[2]]
    max_list = [maximum[0], maximum[1], maximum[2]]
    for idx in range(3):
        if max_list[idx] < min_list[idx]:
            min_list[idx], max_list[idx] = max_list[idx], min_list[idx]
    return (min_list[0], min_list[1], min_list[2]), (max_list[0], max_list[1], max_list[2])


def _random_unit_quaternion(rng: DeterministicRng) -> Tuple[float, float, float, float]:
    x = rng.range(-1.0, 1.0)
    y = rng.range(-1.0, 1.0)
    z = rng.range(-1.0, 1.0)
    w = rng.range(-1.0, 1.0)
    length_sq = _f32(x * x + y * y + z * z + w * w)
    if length_sq < CMP_EPSILON2:
        return 0.0, 0.0, 0.0, 1.0
    inv_len = _f32(1.0 / math.sqrt(length_sq))
    return _f32(x * inv_len), _f32(y * inv_len), _f32(z * inv_len), _f32(w * inv_len)


def _make_normal(rng: DeterministicRng, normal_tilt: float) -> Vec3:
    if normal_tilt <= 0.0:
        return 0.0, 1.0, 0.0
    tilt = _f32(normal_tilt)
    normal = (
        rng.range(-tilt, tilt),
        _f32(1.0),
        rng.range(-tilt, tilt),
    )
    length_sq = _f32(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
    if length_sq < CMP_EPSILON2:
        return 0.0, 1.0, 0.0
    inv_len = _f32(1.0 / math.sqrt(length_sq))
    return _f32(normal[0] * inv_len), _f32(normal[1] * inv_len), _f32(normal[2] * inv_len)


def _cluster_color(seed: int, cluster_index: int, opacity: float) -> Color4:
    cluster_seed = _u64(seed ^ _u64((cluster_index + 1) * 0xA24BAED4963EE407))
    rng = DeterministicRng(cluster_seed)
    r = _f32(0.2 + rng.next_unit_float() * 0.8)
    g = _f32(0.2 + rng.next_unit_float() * 0.8)
    b = _f32(0.2 + rng.next_unit_float() * 0.8)
    return r, g, b, _f32(opacity)


def _hash_gaussian(hash_value: int, gaussian: Dict[str, object]) -> int:
    hash_value = _hash_vector3(hash_value, gaussian["position"])
    hash_value = _hash_vector3(hash_value, gaussian["scale"])
    hash_value = _hash_float(hash_value, gaussian["area"])
    hash_value = _hash_float(hash_value, gaussian["opacity"])
    hash_value = _hash_quaternion(hash_value, gaussian["rotation"])
    hash_value = _hash_color(hash_value, gaussian["sh_dc"])
    sh_1 = gaussian["sh_1"]
    hash_value = _hash_vector3(hash_value, sh_1[0])
    hash_value = _hash_vector3(hash_value, sh_1[1])
    hash_value = _hash_vector3(hash_value, sh_1[2])
    hash_value = _hash_vector3(hash_value, gaussian["normal"])
    hash_value = _hash_float(hash_value, gaussian["stroke_age"])
    hash_value = _hash_vector2(hash_value, gaussian["brush_axes"])
    hash_value = _hash_u64(hash_value, gaussian["painterly_meta"])
    return hash_value


def _hash_uniform_config(config: Dict[str, object]) -> int:
    hash_value = HASH_BASIS
    hash_value = _hash_u64(hash_value, config["splat_count"])
    hash_value = _hash_u64(hash_value, config["seed"])
    hash_value = _hash_vector3(hash_value, config["position_min"])
    hash_value = _hash_vector3(hash_value, config["position_max"])
    hash_value = _hash_float(hash_value, config["min_scale"])
    hash_value = _hash_float(hash_value, config["max_scale"])
    hash_value = _hash_float(hash_value, config["min_opacity"])
    hash_value = _hash_float(hash_value, config["max_opacity"])
    hash_value = _hash_float(hash_value, config["normal_tilt"])
    hash_value = _hash_bool(hash_value, config["random_rotation"])
    hash_value = _hash_bool(hash_value, config["random_colors"])
    hash_value = _hash_color(hash_value, config["base_color"])
    return hash_value


def _hash_clustered_config(config: Dict[str, object]) -> int:
    hash_value = HASH_BASIS
    hash_value = _hash_u64(hash_value, config["splat_count"])
    hash_value = _hash_u64(hash_value, config["seed"])
    hash_value = _hash_u64(hash_value, config["cluster_count"])
    hash_value = _hash_vector3(hash_value, config["cluster_center_min"])
    hash_value = _hash_vector3(hash_value, config["cluster_center_max"])
    hash_value = _hash_vector3(hash_value, config["center_offset"])
    hash_value = _hash_float(hash_value, config["cluster_radius"])
    hash_value = _hash_float(hash_value, config["min_scale"])
    hash_value = _hash_float(hash_value, config["max_scale"])
    hash_value = _hash_float(hash_value, config["min_opacity"])
    hash_value = _hash_float(hash_value, config["max_opacity"])
    hash_value = _hash_float(hash_value, config["normal_tilt"])
    hash_value = _hash_bool(hash_value, config["random_rotation"])
    hash_value = _hash_bool(hash_value, config["color_per_cluster"])
    return hash_value


def _serialize_vec3(value: Vec3) -> List[float]:
    return [_f32(value[0]), _f32(value[1]), _f32(value[2])]


def _serialize_color(value: Color4) -> List[float]:
    return [_f32(value[0]), _f32(value[1]), _f32(value[2]), _f32(value[3])]


def _summarize_scene(splats: Sequence[Dict[str, object]], generator_name: str, seed: int, config_hash: int) -> Dict[str, object]:
    scene_hash = HASH_BASIS
    scene_hash = _hash_u64(scene_hash, seed)
    scene_hash = _hash_u64(scene_hash, config_hash)
    scene_hash = _hash_u64(scene_hash, len(splats))

    if not splats:
        return {
            "generator": generator_name,
            "splat_count": 0,
            "seed": seed,
            "config_hash": config_hash,
            "scene_hash": scene_hash,
            "bounds_min": (0.0, 0.0, 0.0),
            "bounds_max": (0.0, 0.0, 0.0),
            "average_scale": 0.0,
            "average_opacity": 0.0,
        }

    bounds_min = list(splats[0]["position"])
    bounds_max = list(splats[0]["position"])
    scale_accum = 0.0
    opacity_accum = 0.0

    for gaussian in splats:
        position = gaussian["position"]
        bounds_min[0] = min(bounds_min[0], position[0])
        bounds_min[1] = min(bounds_min[1], position[1])
        bounds_min[2] = min(bounds_min[2], position[2])
        bounds_max[0] = max(bounds_max[0], position[0])
        bounds_max[1] = max(bounds_max[1], position[1])
        bounds_max[2] = max(bounds_max[2], position[2])
        scale = gaussian["scale"]
        scale_accum += (scale[0] + scale[1] + scale[2]) / 3.0
        opacity_accum += gaussian["opacity"]
        scene_hash = _hash_gaussian(scene_hash, gaussian)

    return {
        "generator": generator_name,
        "splat_count": len(splats),
        "seed": seed,
        "config_hash": config_hash,
        "scene_hash": scene_hash,
        "bounds_min": (bounds_min[0], bounds_min[1], bounds_min[2]),
        "bounds_max": (bounds_max[0], bounds_max[1], bounds_max[2]),
        "average_scale": _f32(scale_accum / float(len(splats))),
        "average_opacity": _f32(opacity_accum / float(len(splats))),
    }


def _generate_uniform(config: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    normalized = dict(config)
    normalized["position_min"], normalized["position_max"] = _normalize_range_vec3(
        normalized["position_min"], normalized["position_max"]
    )
    normalized["min_scale"], normalized["max_scale"] = _normalize_range(
        normalized["min_scale"], normalized["max_scale"]
    )
    normalized["min_opacity"], normalized["max_opacity"] = _normalize_range(
        normalized["min_opacity"], normalized["max_opacity"]
    )

    rng = DeterministicRng(normalized["seed"])
    splats: List[Dict[str, object]] = []

    for _ in range(normalized["splat_count"]):
        position = (
            rng.range(normalized["position_min"][0], normalized["position_max"][0]),
            rng.range(normalized["position_min"][1], normalized["position_max"][1]),
            rng.range(normalized["position_min"][2], normalized["position_max"][2]),
        )

        scale_value = rng.range(normalized["min_scale"], normalized["max_scale"])
        scale = (scale_value, scale_value, scale_value)
        area = _f32(scale_value * scale_value * math.pi)
        opacity = rng.range(normalized["min_opacity"], normalized["max_opacity"])
        normal = _make_normal(rng, normalized["normal_tilt"])
        rotation: Quat4 = (0.0, 0.0, 0.0, 1.0)

        if normalized["random_rotation"]:
            rotation = _random_unit_quaternion(rng)

        if normalized["random_colors"]:
            sh_dc = (rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), opacity)
        else:
            base_color = normalized["base_color"]
            sh_dc = (base_color[0], base_color[1], base_color[2], opacity)

        splats.append({
            "position": position,
            "scale": scale,
            "area": area,
            "opacity": opacity,
            "rotation": rotation,
            "sh_dc": sh_dc,
            "sh_1": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "normal": normal,
            "stroke_age": 0.0,
            "brush_axes": (1.0, 0.0),
            "painterly_meta": 0,
        })

    config_hash = _hash_uniform_config(normalized)
    summary = _summarize_scene(splats, "uniform", normalized["seed"], config_hash)
    return normalized, summary, splats


def _generate_clustered(config: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    normalized = dict(config)
    normalized["cluster_count"] = max(1, int(normalized["cluster_count"]))
    normalized["cluster_center_min"], normalized["cluster_center_max"] = _normalize_range_vec3(
        normalized["cluster_center_min"], normalized["cluster_center_max"]
    )
    normalized["min_scale"], normalized["max_scale"] = _normalize_range(
        normalized["min_scale"], normalized["max_scale"]
    )
    normalized["min_opacity"], normalized["max_opacity"] = _normalize_range(
        normalized["min_opacity"], normalized["max_opacity"]
    )
    if normalized["cluster_radius"] < 0.0:
        normalized["cluster_radius"] = 0.0

    rng = DeterministicRng(normalized["seed"])
    centers: List[Vec3] = []
    for _ in range(normalized["cluster_count"]):
        centers.append((
            _f32(rng.range(normalized["cluster_center_min"][0], normalized["cluster_center_max"][0]) + normalized["center_offset"][0]),
            _f32(rng.range(normalized["cluster_center_min"][1], normalized["cluster_center_max"][1]) + normalized["center_offset"][1]),
            _f32(rng.range(normalized["cluster_center_min"][2], normalized["cluster_center_max"][2]) + normalized["center_offset"][2]),
        ))

    splats: List[Dict[str, object]] = []
    for index in range(normalized["splat_count"]):
        cluster_idx = index % normalized["cluster_count"]
        center = centers[cluster_idx]
        radius = normalized["cluster_radius"]

        position = (
            _f32(center[0] + rng.range(-radius, radius)),
            _f32(center[1] + rng.range(-radius, radius)),
            _f32(center[2] + rng.range(-radius, radius)),
        )
        scale_value = rng.range(normalized["min_scale"], normalized["max_scale"])
        scale = (scale_value, scale_value, scale_value)
        area = _f32(scale_value * scale_value * math.pi)
        opacity = rng.range(normalized["min_opacity"], normalized["max_opacity"])
        normal = _make_normal(rng, normalized["normal_tilt"])
        rotation: Quat4 = (0.0, 0.0, 0.0, 1.0)

        if normalized["random_rotation"]:
            rotation = _random_unit_quaternion(rng)

        if normalized["color_per_cluster"]:
            sh_dc = _cluster_color(normalized["seed"], cluster_idx, opacity)
        else:
            sh_dc = (rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), opacity)

        splats.append({
            "position": position,
            "scale": scale,
            "area": area,
            "opacity": opacity,
            "rotation": rotation,
            "sh_dc": sh_dc,
            "sh_1": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "normal": normal,
            "stroke_age": 0.0,
            "brush_axes": (1.0, 0.0),
            "painterly_meta": 0,
        })

    config_hash = _hash_clustered_config(normalized)
    summary = _summarize_scene(splats, "clustered", normalized["seed"], config_hash)
    return normalized, summary, splats


def _to_json_ready_config(config: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, tuple) and len(value) == 3:
            output[key] = _serialize_vec3(value)
        elif isinstance(value, tuple) and len(value) == 4:
            output[key] = _serialize_color(value)
        elif isinstance(value, float):
            output[key] = _f32(value)
        else:
            output[key] = value
    return output


def _to_json_ready_summary(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "generator": summary["generator"],
        "splat_count": summary["splat_count"],
        "seed": summary["seed"],
        "config_hash": _hex_u64(summary["config_hash"]),
        "scene_hash": _hex_u64(summary["scene_hash"]),
        "bounds_min": _serialize_vec3(summary["bounds_min"]),
        "bounds_max": _serialize_vec3(summary["bounds_max"]),
        "average_scale": _f32(summary["average_scale"]),
        "average_opacity": _f32(summary["average_opacity"]),
    }


def _preview_splats(splats: Sequence[Dict[str, object]], count: int = 3) -> List[Dict[str, object]]:
    preview: List[Dict[str, object]] = []
    for gaussian in splats[:count]:
        preview.append({
            "position": _serialize_vec3(gaussian["position"]),
            "scale": _serialize_vec3(gaussian["scale"]),
            "opacity": _f32(gaussian["opacity"]),
            "sh_dc": _serialize_color(gaussian["sh_dc"]),
            "normal": _serialize_vec3(gaussian["normal"]),
        })
    return preview


def _build_baselines() -> Dict[str, Dict[str, object]]:
    uniform_raw_config: Dict[str, object] = {
        "splat_count": 256,
        "seed": 42,
        "position_min": (-3.0, -2.0, -1.0),
        "position_max": (3.0, 2.0, 1.0),
        "min_scale": 0.1,
        "max_scale": 0.4,
        "min_opacity": 0.3,
        "max_opacity": 0.9,
        "normal_tilt": 0.2,
        "random_rotation": True,
        "random_colors": True,
        "base_color": (0.7, 0.7, 0.7, 1.0),
    }
    uniform_config, uniform_summary, uniform_splats = _generate_uniform(uniform_raw_config)

    clustered_raw_config: Dict[str, object] = {
        "splat_count": 256,
        "seed": 84,
        "cluster_count": 8,
        "cluster_center_min": (-10.0, -5.0, -2.0),
        "cluster_center_max": (10.0, 5.0, 2.0),
        "center_offset": (0.5, 0.0, 0.0),
        "cluster_radius": 1.5,
        "min_scale": 0.05,
        "max_scale": 0.25,
        "min_opacity": 0.5,
        "max_opacity": 1.0,
        "normal_tilt": 0.0,
        "random_rotation": False,
        "color_per_cluster": True,
    }
    clustered_config, clustered_summary, clustered_splats = _generate_clustered(clustered_raw_config)

    base_contract = {
        "rng": "splitmix64",
        "hash": "fnv1a-64",
        "hash_quantization_scale": HASH_QUANTIZATION_SCALE,
        "note": "Any seed/config change must change at least one hash value.",
    }

    return {
        "uniform_seed_42.json": {
            "artifact_version": 1,
            "generator": "uniform",
            "deterministic_contract": {
                **base_contract,
                "seed": uniform_summary["seed"],
                "splat_count": uniform_summary["splat_count"],
                "config_hash": _hex_u64(uniform_summary["config_hash"]),
                "scene_hash": _hex_u64(uniform_summary["scene_hash"]),
            },
            "raw_config": _to_json_ready_config(uniform_raw_config),
            "normalized_config": _to_json_ready_config(uniform_config),
            "summary": _to_json_ready_summary(uniform_summary),
            "preview_splats": _preview_splats(uniform_splats),
        },
        "clustered_seed_84.json": {
            "artifact_version": 1,
            "generator": "clustered",
            "deterministic_contract": {
                **base_contract,
                "seed": clustered_summary["seed"],
                "splat_count": clustered_summary["splat_count"],
                "config_hash": _hex_u64(clustered_summary["config_hash"]),
                "scene_hash": _hex_u64(clustered_summary["scene_hash"]),
            },
            "raw_config": _to_json_ready_config(clustered_raw_config),
            "normalized_config": _to_json_ready_config(clustered_config),
            "summary": _to_json_ready_summary(clustered_summary),
            "preview_splats": _preview_splats(clustered_splats),
        },
    }


def _as_json(data: Dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _write_baselines(output_dir: Path, baselines: Dict[str, Dict[str, object]]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, content in baselines.items():
        destination = output_dir / file_name
        destination.write_text(_as_json(content), encoding="utf-8")
        print(f"Wrote {destination}")
    return 0


def _check_baselines(output_dir: Path, baselines: Dict[str, Dict[str, object]]) -> int:
    failed = False
    for file_name, expected_content in baselines.items():
        destination = output_dir / file_name
        if not destination.exists():
            print(f"Missing baseline artifact: {destination}")
            failed = True
            continue

        actual = destination.read_text(encoding="utf-8")
        expected = _as_json(expected_content)
        if actual != expected:
            print(f"Outdated baseline artifact: {destination}")
            failed = True
        else:
            print(f"OK: {destination}")

    extra_files = sorted(path.name for path in output_dir.glob("*.json"))
    expected_files = sorted(baselines.keys())
    unexpected = [name for name in extra_files if name not in expected_files]
    if unexpected:
        print(f"Unexpected baseline artifacts in {output_dir}: {', '.join(unexpected)}")
        failed = True

    if failed:
        print("Synthetic baseline check failed. Re-run without --check to update artifacts.")
        return 1

    print("Synthetic baseline check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic splat baseline artifacts")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for baseline JSON artifacts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed artifacts match deterministic generation output",
    )
    args = parser.parse_args()

    baselines = _build_baselines()
    output_dir = Path(args.output_dir).resolve()
    if args.check:
        return _check_baselines(output_dir, baselines)
    return _write_baselines(output_dir, baselines)


if __name__ == "__main__":
    raise SystemExit(main())

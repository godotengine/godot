#!/usr/bin/env python3
"""Canonical staging contract for benchmark-side chunked open-world assets."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

CHUNK_SPLAT_COUNT = 65_536
MAIN_PROJECT_FIXTURE_ROOT = "tests/examples/godot/test_project/tests/fixtures/open_world"
MAIN_PROJECT_RES_ROOT = "res://tests/fixtures/open_world"
STAGE_MANIFEST_SUFFIX = ".stage_manifest.json"
CHUNKED_LADDER_REF_PREFIX = "chunked_ladder:"
WORLD_SOURCE_ASSET_PATH = "res://tests/fixtures/synthetic_spiral.ply"
WORLD_SOURCE_ASSET_SPLATS = 25_000


@dataclass(frozen=True)
class OpenWorldAssetSpec:
    asset_id: str
    topology: str
    total_splats: int
    target_visible_chunks: int
    target_resident_chunks: int
    seed: int
    notes: str

    @property
    def chunk_splats(self) -> int:
        return CHUNK_SPLAT_COUNT

    @property
    def chunk_count(self) -> int:
        return int(math.ceil(float(self.total_splats) / float(self.chunk_splats)))

    @property
    def max_visible_splats(self) -> int:
        return self.target_visible_chunks * self.chunk_splats

    @property
    def max_resident_splats(self) -> int:
        return self.target_resident_chunks * self.chunk_splats


OPEN_WORLD_ASSET_SPECS: tuple[OpenWorldAssetSpec, ...] = (
    OpenWorldAssetSpec(
        asset_id="open_world_corridor_20m",
        topology="corridor_return",
        total_splats=20_000_000,
        target_visible_chunks=24,
        target_resident_chunks=48,
        seed=42020,
        notes=(
            "Corridor-style open-world proof asset. Total scale is 20M splats across many chunks, "
            "while the intended visible window stays bounded to a narrow return-path corridor."
        ),
    ),
    OpenWorldAssetSpec(
        asset_id="open_world_boundary_50m",
        topology="biome_boundary_crossing",
        total_splats=50_000_000,
        target_visible_chunks=32,
        target_resident_chunks=80,
        seed=45050,
        notes=(
            "Boundary-crossing proof asset. Intended for large visibility shifts and forward/reverse "
            "crossing without collapsing into a monolithic single-file benchmark shortcut."
        ),
    ),
    OpenWorldAssetSpec(
        asset_id="open_world_city_100m",
        topology="city_block_roam_soak",
        total_splats=100_000_000,
        target_visible_chunks=48,
        target_resident_chunks=128,
        seed=50100,
        notes=(
            "City-scale roam/soak proof asset. Total scale reaches 100M splats, but the intended "
            "resident and visible working sets remain explicitly bounded."
        ),
    ),
)


def build_chunked_asset_ladder() -> dict[str, dict[str, object]]:
    return {spec.asset_id: _build_asset_entry(spec) for spec in OPEN_WORLD_ASSET_SPECS}


def build_chunked_asset_reference(asset_id: str) -> str:
    return f"{CHUNKED_LADDER_REF_PREFIX}{asset_id}"


def validate_chunked_asset_ladder(ladder: dict[str, dict[str, object]] | None = None) -> list[str]:
    expected = ladder or build_chunked_asset_ladder()
    failures: list[str] = []
    seen_asset_ids: set[str] = set()
    seen_stage_paths: set[str] = set()

    for spec in OPEN_WORLD_ASSET_SPECS:
        entry = expected.get(spec.asset_id)
        if entry is None:
            failures.append(f"missing asset ladder entry for {spec.asset_id}")
            continue

        seen_asset_ids.add(spec.asset_id)
        contract = entry.get("working_set_contract", {})
        staging = entry.get("staging", {})
        generation = entry.get("generation", {})
        bootstrap_builder = entry.get("bootstrap_world_builder", {})

        if entry.get("asset_classification") != "chunked_open_world_candidate":
            failures.append(f"{spec.asset_id}: asset_classification must be chunked_open_world_candidate")
        if entry.get("evidence_role") != "open_world_proof_pending_staging":
            failures.append(f"{spec.asset_id}: evidence_role must be open_world_proof_pending_staging")
        if entry.get("staging_status") != "planned_unstaged":
            failures.append(f"{spec.asset_id}: staging_status must be planned_unstaged")
        if contract.get("chunk_splats") != CHUNK_SPLAT_COUNT:
            failures.append(f"{spec.asset_id}: chunk_splats must be {CHUNK_SPLAT_COUNT}")
        if contract.get("chunk_count") != spec.chunk_count:
            failures.append(f"{spec.asset_id}: chunk_count must be {spec.chunk_count}")
        if contract.get("max_visible_chunks") != spec.target_visible_chunks:
            failures.append(f"{spec.asset_id}: max_visible_chunks must be {spec.target_visible_chunks}")
        if contract.get("max_resident_chunks") != spec.target_resident_chunks:
            failures.append(f"{spec.asset_id}: max_resident_chunks must be {spec.target_resident_chunks}")
        if generation.get("seed") != spec.seed:
            failures.append(f"{spec.asset_id}: seed must be {spec.seed}")
        if not isinstance(bootstrap_builder, dict):
            failures.append(f"{spec.asset_id}: bootstrap_world_builder must be an object")
        elif str(bootstrap_builder.get("source_asset_path", "")).strip() != WORLD_SOURCE_ASSET_PATH:
            failures.append(f"{spec.asset_id}: source_asset_path must be {WORLD_SOURCE_ASSET_PATH}")
        elif int(bootstrap_builder.get("source_asset_splats", 0)) != WORLD_SOURCE_ASSET_SPLATS:
            failures.append(f"{spec.asset_id}: source_asset_splats must be {WORLD_SOURCE_ASSET_SPLATS}")
        elif int(bootstrap_builder.get("instance_count", 0)) <= 0:
            failures.append(f"{spec.asset_id}: bootstrap instance_count must be > 0")
        elif int(bootstrap_builder.get("materialized_total_splats", 0)) != spec.total_splats:
            failures.append(f"{spec.asset_id}: materialized_total_splats must be {spec.total_splats}")
        elif int(bootstrap_builder.get("instance_count", 0)) * int(bootstrap_builder.get("source_asset_splats", 0)) != spec.total_splats:
            failures.append(f"{spec.asset_id}: instance_count * source_asset_splats must equal {spec.total_splats}")

        stage_manifest_path = str(staging.get("project_stage_manifest_path", ""))
        if not stage_manifest_path:
            failures.append(f"{spec.asset_id}: missing project_stage_manifest_path")
        elif stage_manifest_path in seen_stage_paths:
            failures.append(f"{spec.asset_id}: duplicate stage manifest path {stage_manifest_path}")
        else:
            seen_stage_paths.add(stage_manifest_path)

    unexpected_asset_ids = sorted(set(expected.keys()) - seen_asset_ids)
    if unexpected_asset_ids:
        failures.append(f"unexpected asset ladder ids: {', '.join(unexpected_asset_ids)}")
    return failures


def write_stage_manifests(output_root: Path) -> list[Path]:
    written: list[Path] = []
    for spec in OPEN_WORLD_ASSET_SPECS:
        entry = _build_asset_entry(spec)
        manifest_path = output_root / spec.asset_id / f"{spec.asset_id}{STAGE_MANIFEST_SUFFIX}"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(manifest_path)
    return written


def _build_asset_entry(spec: OpenWorldAssetSpec) -> dict[str, object]:
    stage_manifest_name = f"{spec.asset_id}{STAGE_MANIFEST_SUFFIX}"
    project_stage_manifest_path = f"{MAIN_PROJECT_RES_ROOT}/{spec.asset_id}/{stage_manifest_name}"
    repo_stage_manifest_path = f"{MAIN_PROJECT_FIXTURE_ROOT}/{spec.asset_id}/{stage_manifest_name}"
    working_set_fraction = round(float(spec.max_visible_splats) / float(spec.total_splats), 6)
    resident_fraction = round(float(spec.max_resident_splats) / float(spec.total_splats), 6)

    return {
        "asset_classification": "chunked_open_world_candidate",
        "asset_id": spec.asset_id,
        "bootstrap_world_builder": _build_bootstrap_world_builder(spec),
        "evidence_role": "open_world_proof_pending_staging",
        "generation": {
            "chunk_naming_pattern": "chunks/chunk_{chunk_index:04d}.ply",
            "helper_script": "tests/runtime/open_world_chunked_asset_ladder.py",
            "materialization_support": "runtime_world_builder_contract",
            "seed": spec.seed,
            "topology": spec.topology,
        },
        "notes": spec.notes,
        "staging": {
            "project_fixture_root": f"{MAIN_PROJECT_FIXTURE_ROOT}/{spec.asset_id}",
            "project_res_root": f"{MAIN_PROJECT_RES_ROOT}/{spec.asset_id}",
            "project_stage_manifest_path": project_stage_manifest_path,
            "repo_stage_manifest_path": repo_stage_manifest_path,
        },
        "staging_status": "planned_unstaged",
        "working_set_contract": {
            "chunk_count": spec.chunk_count,
            "chunk_splats": spec.chunk_splats,
            "max_resident_chunks": spec.target_resident_chunks,
            "max_resident_splats": spec.max_resident_splats,
            "max_visible_chunks": spec.target_visible_chunks,
            "max_visible_fraction": working_set_fraction,
            "max_visible_splats": spec.max_visible_splats,
            "resident_fraction": resident_fraction,
            "total_splats": spec.total_splats,
        },
    }


def _build_bootstrap_world_builder(spec: OpenWorldAssetSpec) -> dict[str, object]:
    topology_defaults = {
        "corridor_return": {
            "builder_kind": "corridor_world_bootstrap",
            "corridor_lanes": 8,
            "corridor_segments": 100,
            "lane_spacing": 7.5,
            "segment_spacing": 10.0,
            "camera_path_hint": "corridor_return",
        },
        "biome_boundary_crossing": {
            "builder_kind": "boundary_world_bootstrap",
            "corridor_lanes": 20,
            "corridor_segments": 100,
            "lane_spacing": 12.0,
            "segment_spacing": 12.0,
            "camera_path_hint": "boundary_crossing",
        },
        "city_block_roam_soak": {
            "builder_kind": "city_world_bootstrap",
            "corridor_lanes": 20,
            "corridor_segments": 200,
            "lane_spacing": 14.0,
            "segment_spacing": 14.0,
            "camera_path_hint": "city_roam",
        },
    }
    defaults = topology_defaults.get(spec.topology, topology_defaults["corridor_return"])
    instance_count = int(defaults["corridor_lanes"]) * int(defaults["corridor_segments"])
    return {
        "materialized_total_splats": instance_count * WORLD_SOURCE_ASSET_SPLATS,
        "chunk_size": 10.0,
        "corridor_lanes": int(defaults["corridor_lanes"]),
        "corridor_segments": int(defaults["corridor_segments"]),
        "lane_spacing": float(defaults["lane_spacing"]),
        "segment_spacing": float(defaults["segment_spacing"]),
        "camera_path_hint": str(defaults["camera_path_hint"]),
        "instance_count": instance_count,
        "source_asset_path": WORLD_SOURCE_ASSET_PATH,
        "source_asset_splats": WORLD_SOURCE_ASSET_SPLATS,
        "world_resource_kind": "gaussian_world_contract",
        "builder_kind": str(defaults["builder_kind"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or stage the canonical open-world chunked asset ladder.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the built-in open-world chunked asset ladder contract.",
    )
    parser.add_argument(
        "--write-stage-manifests",
        default="",
        help="Write metadata-only stage manifests under the given output directory.",
    )
    args = parser.parse_args()

    failures = validate_chunked_asset_ladder()
    if failures:
        print("[open-world-asset-ladder] validation failed")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if args.write_stage_manifests:
        output_root = Path(args.write_stage_manifests).expanduser().resolve()
        written = write_stage_manifests(output_root)
        print(f"[open-world-asset-ladder] wrote {len(written)} stage manifests under {output_root}")
        return 0

    if args.check:
        print("[open-world-asset-ladder] validation passed")
        return 0

    print(json.dumps(build_chunked_asset_ladder(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fixture preview validator for Gaussian splats (no Godot required).

We rasterise the checked-in PLY fixture onto a small RGB grid, compare the bytes
against a stored reference, and report the maximum per-channel deviation.  The
reference lives in `visual_quality_reference.json` so the check remains
deterministic without generating large artefacts or depending on Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "templates"
    / "gaussian_splat_template"
    / "assets"
    / "template_splats.ply"
)
REFERENCE_PATH = Path(__file__).parent / "visual_quality_reference.json"


def _parse_vertices(path: Path) -> List[Sequence[float]]:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    count = 0
    properties: List[str] = []
    start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            count = int(stripped.split()[-1])
        elif stripped.startswith("property "):
            properties.append(stripped.split()[-1])
        elif stripped == "end_header":
            start = idx + 1
            break
    else:  # pragma: no cover - defensive
        raise ValueError("PLY header missing 'end_header'.")

    prop_index = {name: i for i, name in enumerate(properties)}
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    for prop in required:
        if prop not in prop_index:
            raise ValueError(f"PLY file missing required property '{prop}'.")

    vertices: List[Sequence[float]] = []
    for raw in lines[start : start + count]:
        values = [float(item) for item in raw.strip().split()]
        vertices.append(values)
    return vertices, prop_index


def _norm(value: float, bounds: Tuple[float, float]) -> float:
    minimum, maximum = bounds
    if maximum == minimum:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def rasterise_fixture(path: Path, size: int = 64) -> bytearray:
    vertices, prop_index = _parse_vertices(path)
    coords = [(v[prop_index["x"]], v[prop_index["y"]], v[prop_index["z"]]) for v in vertices]
    colors = [(v[prop_index["f_dc_0"]], v[prop_index["f_dc_1"]], v[prop_index["f_dc_2"]]) for v in vertices]

    xs, ys, zs = zip(*coords)
    bounds_x = (min(xs), max(xs))
    bounds_y = (min(ys), max(ys))
    bounds_z = (min(zs), max(zs))

    pixels = bytearray(size * size * 3)

    for (x, y, z), (r, g, b) in zip(coords, colors):
        u = int(_norm(x, bounds_x) * (size - 1))
        v = int(_norm(y, bounds_y) * (size - 1))
        depth = _norm(z, bounds_z)

        rb = int(max(0.0, min(1.0, r)) * 255)
        gb = int(max(0.0, min(1.0, g)) * 255)
        bb = int(max(0.0, min(1.0, b)) * 255)

        # Simple depth shading (closer splats appear brighter).
        brightness = 0.6 + depth * 0.4
        colour = (min(255, int(rb * brightness)), min(255, int(gb * brightness)), min(255, int(bb * brightness)))

        # Flip Y so positive values appear near the top.
        idx = ((size - 1 - v) * size + u) * 3
        pixels[idx : idx + 3] = bytes(colour)

    return pixels


@dataclass
class VisualResult:
    matches_reference: bool
    max_channel_delta: int
    hash: str

    def to_dict(self) -> dict:
        return {
            "matches_reference": self.matches_reference,
            "max_channel_delta": self.max_channel_delta,
            "SHA256": self.hash,
        }


def _max_delta(a: Iterable[int], b: Iterable[int]) -> int:
    return max(abs(x - y) for x, y in zip(a, b))


def compare_to_reference(pixels: bytearray, reference: dict) -> VisualResult:
    reference_bytes = bytes.fromhex(reference["pixels_hex"])
    if len(reference_bytes) != len(pixels):
        raise ValueError("Reference preview size does not match current render.")
    max_delta = _max_delta(pixels, reference_bytes)
    digest = hashlib.sha256(pixels).hexdigest()
    return VisualResult(matches_reference=max_delta == 0, max_channel_delta=max_delta, hash=digest)


def write_reference(size: int, pixels: bytearray) -> None:
    data = {
        "format": "rgb8",
        "size": size,
        "pixels_hex": bytes(pixels).hex(),
    }
    REFERENCE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[visual] Updated reference at {REFERENCE_PATH.relative_to(Path.cwd())}")


def load_reference() -> dict | None:
    if not REFERENCE_PATH.exists():
        return None
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def save_report(result: VisualResult) -> None:
    report = {
        "fixture": str(FIXTURE_PATH.relative_to(Path(__file__).parents[1])),
        "result": result.to_dict(),
    }
    report_path = Path(__file__).parent / "visual_quality_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[visual] Report written to {report_path.relative_to(Path.cwd())}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gaussian splat fixture preview.")
    parser.add_argument("--preview-size", type=int, default=64, help="Raster size used for the comparison grid.")
    parser.add_argument("--update-reference", action="store_true", help="Refresh the stored reference data.")
    args = parser.parse_args()

    if not FIXTURE_PATH.exists():
        raise SystemExit(f"Fixture not found: {FIXTURE_PATH}")

    pixels = rasterise_fixture(FIXTURE_PATH, size=args.preview_size)
    if args.update_reference or not REFERENCE_PATH.exists():
        write_reference(args.preview_size, pixels)
        reference = {"size": args.preview_size, "pixels_hex": bytes(pixels).hex()}
    else:
        reference = load_reference()
        if reference is None:
            raise SystemExit("Reference file could not be parsed.")

    result = compare_to_reference(pixels, reference)
    status = "MATCH" if result.matches_reference else f"DRIFT (Δ={result.max_channel_delta})"
    print(f"[visual] Comparison result: {status} | SHA256={result.hash}")

    save_report(result)
    return 0 if result.matches_reference else 1


if __name__ == "__main__":
    raise SystemExit(main())

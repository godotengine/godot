#!/usr/bin/env python3
"""Lightweight performance probes for the Gaussian splat fixtures.

The original Phase 4 benchmark tried to exercise a full Godot build, generate
million-splat assets, and collect dozens of synthetic metrics.  Those steps are
impractical for day-to-day development, so this trimmed-down version focuses on
repeatable CPU-side checks that still provide signal when the fixture data
changes.

The script parses the small PLY fixtures that back our tests, computes a few
basic statistics (bounding boxes, centroid distance, naive depth-sort cost),
and saves the results to `benchmark_results.json`.  The numbers are quick to
generate (<1 ms on a laptop) yet still catch accidental regressions such as
empty fixtures or corrupted coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
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
REPORT_PATH = Path(__file__).parent / "benchmark_results.json"


def _parse_header(lines: Iterable[str]) -> Tuple[int, Sequence[str]]:
    count = 0
    properties: List[str] = []
    header_finished = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "end_header":
            header_finished = True
            break
        if stripped.startswith("element vertex"):
            _, _, count_str = stripped.split()
            count = int(count_str)
        elif stripped.startswith("property "):
            parts = stripped.split()
            properties.append(parts[-1])
    if not header_finished:
        raise ValueError("PLY header did not terminate with 'end_header'")
    return count, properties


def _iter_vertices(path: Path) -> Iterable[List[float]]:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    count, properties = _parse_header(lines)
    start = lines.index("end_header\n") + 1
    for idx in range(count):
        raw = lines[start + idx].strip()
        if not raw:
            continue
        yield [float(value) for value in raw.split()]


@dataclass(frozen=True)
class FixtureMetrics:
    splat_count: int
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    mean_distance: float
    depth_sort_ms: float

    def to_dict(self) -> dict:
        return {
            "splat_count": self.splat_count,
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
            "mean_distance": self.mean_distance,
            "depth_sort_ms": self.depth_sort_ms,
        }


def analyse_fixture(path: Path) -> FixtureMetrics:
    vertices = list(_iter_vertices(path))
    if not vertices:
        raise ValueError(f"Fixture '{path}' does not contain any vertices.")

    positions = [vertex[:3] for vertex in vertices]
    xs, ys, zs = zip(*positions)
    bounds_min = (min(xs), min(ys), min(zs))
    bounds_max = (max(xs), max(ys), max(zs))

    distances = [math.sqrt(x * x + y * y + z * z) for x, y, z in positions]
    mean_distance = statistics.fmean(distances)

    # Measure a simple depth sort so we notice order-of-magnitude regressions.
    start = time.perf_counter()
    for _ in range(64):
        sorted(positions, key=lambda pos: pos[2])
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 64.0

    return FixtureMetrics(
        splat_count=len(positions),
        bounds_min=tuple(bounds_min),
        bounds_max=tuple(bounds_max),
        mean_distance=mean_distance,
        depth_sort_ms=elapsed_ms,
    )


def write_report(metrics: FixtureMetrics) -> None:
    payload = {
        "fixture": str(FIXTURE_PATH.relative_to(Path(__file__).parents[1])),
        "metrics": metrics.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] Wrote {REPORT_PATH.relative_to(Path.cwd())}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse Gaussian splat fixtures.")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH, help="Override the PLY fixture path.")
    args = parser.parse_args()

    fixture = args.fixture.resolve()
    if not fixture.exists():
        raise SystemExit(f"Fixture not found: {fixture}")

    metrics = analyse_fixture(fixture)
    write_report(metrics)

    print(
        f"[bench] {metrics.splat_count} splats | bounds {metrics.bounds_min} -> {metrics.bounds_max} "
        f"| mean radius {metrics.mean_distance:.3f} | depth sort {metrics.depth_sort_ms:.4f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

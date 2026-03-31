#!/usr/bin/env python3
"""Export benchmark suite data to Vega-Lite JSON and CSV formats for the public performance dashboard."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT_PATHS = [
    ROOT / "tests" / "output" / "benchmark_run" / "benchmark_suite_report.json",
]

EXPORT_FIELDS = [
    "lane_id",
    "lane_name",
    "score",
    "avg_fps",
    "p99_frame_ms",
    "gpu_time_frame_ms",
    "capture_ssim_min",
    "capture_psnr_min",
    "weight",
]


def _safe_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        metric = float(value)
        if math.isfinite(metric):
            return metric
    return None


def find_report(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.is_file() else None
    for candidate in DEFAULT_REPORT_PATHS:
        if candidate.is_file():
            return candidate
    return None


def extract_lanes(report: dict[str, Any]) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    for result in report.get("lane_results", []):
        row: dict[str, Any] = {}
        for field in EXPORT_FIELDS:
            value = result.get(field)
            if field in ("lane_id", "lane_name"):
                row[field] = str(value) if value is not None else ""
            else:
                row[field] = _safe_float(value)
        if row.get("score") is not None or row.get("avg_fps") is not None:
            lanes.append(row)
    return lanes


def write_json(lanes: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the public snapshot a direct reflection of the benchmark report rows.
    output.write_text(json.dumps(lanes, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark-export] Wrote {len(lanes)} lanes to {output}")


def write_csv(lanes: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for lane in lanes:
            writer.writerow({k: ("" if v is None else v) for k, v in lane.items()})
    print(f"[benchmark-export] Wrote {len(lanes)} lanes to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark data for Vega-Lite charts.")
    parser.add_argument(
        "--suite-report",
        type=Path,
        default=None,
        help="Path to benchmark_suite_report.json. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "docs" / "assets" / "data" / "benchmark_latest.json",
        help="Output JSON file for Vega-Lite.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "outputs" / "performance" / "benchmark_latest.csv",
        help="Output CSV file for matplotlib.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = find_report(args.suite_report)
    if report_path is None:
        print("[benchmark-export] No benchmark report found; skipping.")
        return 0

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    lanes = extract_lanes(report)
    if not lanes:
        print("[benchmark-export] No valid lane results in report; skipping.")
        return 0

    write_json(lanes, args.json_output)
    write_csv(lanes, args.csv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

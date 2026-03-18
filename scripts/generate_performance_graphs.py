#!/usr/bin/env python3
"""Generate performance graphs from benchmark CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install with pip install -r docs/requirements.txt") from exc


def find_csv(directory: Path, expected_columns: set[str]) -> Optional[pd.DataFrame]:
    for csv_path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if expected_columns.issubset(set(df.columns)):
            return df
    return None


def generate_fps_graph(input_dir: Path, output_dir: Path) -> None:
    expected = {"dataset", "splat_count", "fps", "gpu"}
    df = find_csv(input_dir, expected)
    if df is None:
        print("[perf] No FPS dataset found; skipping FPS graph.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu, group in df.groupby("gpu"):
        ax.plot(group["splat_count"], group["fps"], marker="o", label=gpu)
    ax.set_xlabel("Splat Count")
    ax.set_ylabel("Frames Per Second")
    ax.set_title("Gaussian Splatting Performance")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    output_path = output_dir / "benchmark_fps.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[perf] Wrote FPS graph to {output_path}")


def generate_streaming_graph(input_dir: Path, output_dir: Path) -> None:
    expected = {"time_s", "stall_percent"}
    df = find_csv(input_dir, expected)
    if df is None:
        print("[perf] No streaming dataset found; skipping stall graph.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time_s"], df["stall_percent"], color="tab:red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stall %")
    ax.set_title("Streaming Stall Rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    output_path = output_dir / "streaming_stalls.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[perf] Wrote streaming stall graph to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance graphs.")
    parser.add_argument("--input", type=Path, default=Path("outputs/performance"), help="Directory containing CSV files.")
    parser.add_argument("--output", type=Path, default=Path("docs/performance"), help="Directory for generated images.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    generate_fps_graph(args.input, args.output)
    generate_streaming_graph(args.input, args.output)


if __name__ == "__main__":
    main()

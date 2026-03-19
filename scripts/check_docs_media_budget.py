#!/usr/bin/env python3
"""Validate documentation media size budgets."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MediaFile:
    path: Path
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def collect_media_files(root: Path, patterns: list[str]) -> list[MediaFile]:
    files: list[MediaFile] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                files.append(MediaFile(path=path, size_bytes=path.stat().st_size))
    unique = {item.path.resolve(): item for item in files}
    return sorted(unique.values(), key=lambda item: item.path.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check docs media size budget.")
    parser.add_argument("--root", default="docs", help="Root directory to scan.")
    parser.add_argument("--glob", action="append", default=["*.mp4", "*.webm"], help="Media filename glob pattern.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Maximum allowed single media file size (MiB).")
    parser.add_argument("--max-total-mb", type=float, default=250.0, help="Maximum allowed total media size (MiB).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = (REPO_ROOT / args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Media root does not exist: {root}")

    files = collect_media_files(root, args.glob)
    total_bytes = sum(item.size_bytes for item in files)
    total_mb = total_bytes / (1024 * 1024)
    max_file_bytes = int(args.max_file_mb * 1024 * 1024)
    max_total_bytes = int(args.max_total_mb * 1024 * 1024)

    print(f"[docs-media] Root: {root}")
    print(f"[docs-media] Patterns: {args.glob}")
    print(f"[docs-media] Files found: {len(files)}")
    for item in files:
        rel_path = item.path.relative_to(REPO_ROOT)
        print(f"[docs-media] {rel_path.as_posix()}: {item.size_mb:.2f} MiB")

    failures: list[str] = []
    for item in files:
        if item.size_bytes > max_file_bytes:
            rel_path = item.path.relative_to(REPO_ROOT)
            failures.append(
                f"{rel_path.as_posix()} exceeds per-file budget: {item.size_mb:.2f} MiB > {args.max_file_mb:.2f} MiB"
            )

    if total_bytes > max_total_bytes:
        failures.append(f"Total media budget exceeded: {total_mb:.2f} MiB > {args.max_total_mb:.2f} MiB")

    if failures:
        for failure in failures:
            print(f"[docs-media] ERROR: {failure}")
        return 1

    print(f"[docs-media] Total media size: {total_mb:.2f} MiB (budget: {args.max_total_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

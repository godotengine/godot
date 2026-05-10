#!/usr/bin/env python3

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(command: list[str], cwd: Path, timeout: int) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Expected output file was not created: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a normal rendered frame against an offscreen rendered frame."
    )
    parser.add_argument("--godot-bin", required=True, type=Path, help="Path to a Godot executable.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "godot_offscreen_compare",
        help="Directory where captures and diff output will be written.",
    )
    parser.add_argument("--resolution", default="640x360", help="Viewport resolution used for both captures.")
    parser.add_argument("--rendering-driver", default="vulkan", help="Rendering driver used for both captures.")
    parser.add_argument(
        "--normal-display-driver",
        default="",
        help="Optional display driver for the normal capture. Defaults to the platform display driver.",
    )
    parser.add_argument("--max-channel-delta", type=int, default=0, help="Per-channel tolerance for a pixel.")
    parser.add_argument(
        "--max-different-pixels",
        type=int,
        default=0,
        help="Number of pixels allowed to exceed --max-channel-delta.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each Godot invocation.")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    godot_bin = args.godot_bin.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_png = output_dir / "normal.png"
    offscreen_png = output_dir / "offscreen.png"
    diff_png = output_dir / "diff.png"

    common_capture_args = [
        str(godot_bin),
        "--path",
        str(project_dir),
        "--rendering-driver",
        args.rendering_driver,
        "--resolution",
        args.resolution,
        "--disable-crash-handler",
        "--quit-after",
        "120",
    ]

    normal_command = common_capture_args.copy()
    if args.normal_display_driver:
        normal_command[1:1] = ["--display-driver", args.normal_display_driver]
    normal_command += ["--", "--capture-output", str(normal_png)]

    offscreen_command = common_capture_args.copy()
    offscreen_command[1:1] = ["--offscreen"]
    offscreen_command += ["--", "--capture-output", str(offscreen_png)]

    compare_command = [
        str(godot_bin),
        "--headless",
        "--path",
        str(project_dir),
        "--script",
        "res://compare_images.gd",
        "--",
        "--reference",
        str(normal_png),
        "--candidate",
        str(offscreen_png),
        "--diff",
        str(diff_png),
        "--max-channel-delta",
        str(args.max_channel_delta),
        "--max-different-pixels",
        str(args.max_different_pixels),
    ]

    run_command(normal_command, project_dir, args.timeout)
    require_file(normal_png)
    run_command(offscreen_command, project_dir, args.timeout)
    require_file(offscreen_png)
    run_command(compare_command, project_dir, args.timeout)
    require_file(diff_png)

    print("NORMAL_CAPTURE=", normal_png)
    print("OFFSCREEN_CAPTURE=", offscreen_png)
    print("DIFF_CAPTURE=", diff_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Generate tiny PLY fixtures to isolate projection/math issues.

Modes:
- single: one red gaussian at the origin
- grid: 10x10 grid in XY plane
- depth_stack: three overlapping gaussians at different depths
"""

import argparse
import math
import struct
from pathlib import Path

SH_C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
DEFAULT_SCALE = 0.3
DEFAULT_OPACITY = 0.8


def _ply_header(count: int) -> str:
    return f"""ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""


def _pack_gaussian(x, y, z, color, scale=DEFAULT_SCALE, opacity=DEFAULT_OPACITY, rot=(1.0, 0.0, 0.0, 0.0)):
    # PLY stores opacity in logit and scale in log domain.
    opacity_logit = math.log(opacity / (1.0 - opacity))
    scale_log = math.log(scale)
    r, g, b = (c / SH_C0 for c in color)
    rot_w, rot_x, rot_y, rot_z = rot
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
        float(rot_w),
        float(rot_x),
        float(rot_y),
        float(rot_z),
    )


def make_single():
    return [_pack_gaussian(0.0, 0.0, -5.0, (1.0, 0.0, 0.0), scale=0.5, opacity=0.99)]


def make_grid():
    splats = []
    for y in range(10):
        for x in range(10):
            px = (x - 4.5) * 0.5
            py = (y - 4.5) * 0.5
            color = ((x % 3) == 0, (x % 3) == 1, (x % 3) == 2)
            splats.append(_pack_gaussian(px, py, -4.0, color, scale=0.35, opacity=0.8))
    return splats


def make_depth_stack():
    return [
        _pack_gaussian(0.0, 0.0, -2.5, (1.0, 0.0, 0.0), scale=0.6, opacity=0.9),  # Near red
        _pack_gaussian(0.0, 0.0, -4.0, (0.0, 1.0, 0.0), scale=0.6, opacity=0.9),  # Mid green
        _pack_gaussian(0.0, 0.0, -6.0, (0.0, 0.0, 1.0), scale=0.6, opacity=0.9),  # Far blue
    ]


def write_ply(path: Path, splats):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(_ply_header(len(splats)).encode("ascii"))
        for splat in splats:
            f.write(struct.pack("<14f", *splat))
    print(f"[generate_test_ply] Wrote {len(splats)} splats to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tiny Gaussian PLY fixtures")
    parser.add_argument("mode", choices=["single", "grid", "depth_stack"], help="fixture to generate")
    parser.add_argument("--out", default="test_data", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.mode == "single":
        write_ply(out_dir / "single_gaussian.ply", make_single())
    elif args.mode == "grid":
        write_ply(out_dir / "grid_gaussians.ply", make_grid())
    elif args.mode == "depth_stack":
        write_ply(out_dir / "depth_stack_gaussians.ply", make_depth_stack())


if __name__ == "__main__":
    main()

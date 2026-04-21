#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from methods import download_d3d12

parser = argparse.ArgumentParser(description="Install D3D12 dependencies for Windows platforms.")
parser.add_argument(
    "--mingw_prefix",
    default=os.getenv("MINGW_PREFIX", ""),
    help="Explicitly specify a path containing the MinGW bin folder.",
)
args = parser.parse_args()

archs = [
    "arm64-llvm",
    "arm64-msvc",
    "x86_32-gcc",
    "x86_32-llvm",
    "x86_32-msvc",
    "x86_64-gcc",
    "x86_64-llvm",
    "x86_64-msvc",
]

download_d3d12(args.mingw_prefix, archs)

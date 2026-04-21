#!/usr/bin/env python3

import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from methods import download_angle

archs = [
    "arm64-llvm",
    "x86_32-gcc",
    "x86_32-llvm",
    "x86_64-gcc",
    "x86_64-llvm",
]
if platform.system() == "Windows":
    # Only download MSVC libraries if we can build using it.
    archs.append("arm64-msvc")
    archs.append("x86_32-msvc")
    archs.append("x86_64-msvc")
elif platform.system() == "Darwin":
    # Only download macOS/iOS libraries if we can build for these platforms.
    archs.append("arm64-ios")
    archs.append("arm64-ios-sim")
    archs.append("arm64-macos")
    archs.append("x86_64-macos")

download_angle(archs)

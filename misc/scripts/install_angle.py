#!/usr/bin/env python

import os
import platform
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from misc.utility.color import Ansi, color_print

# Base Godot dependencies path
# If cross-compiling (no LOCALAPPDATA), we install in `bin`
deps_folder = os.getenv("LOCALAPPDATA")
if deps_folder:
    deps_folder = os.path.join(deps_folder, "Godot", "build_deps")
else:
    deps_folder = os.path.join("bin", "build_deps")

# ANGLE
# Check for latest version: https://github.com/godotengine/godot-angle-static/releases/latest
angle_version = "chromium/7219"
angle_folder = os.path.join(deps_folder, "angle")

# Create dependencies folder
if not os.path.exists(deps_folder):
    os.makedirs(deps_folder)

# Mesa NIR
print(f"Downloading ANGLE {angle_version} ...")
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

for arch in archs:
    angle_filename = f"godot-angle-static-{arch}-release.zip"
    angle_archive = os.path.join(deps_folder, angle_filename)
    angle_folder = os.path.join(deps_folder, f"angle-{arch}")

    if os.path.isfile(angle_archive):
        os.remove(angle_archive)
    print(f"Downloading ANGLE {angle_filename} ...")
    urllib.request.urlretrieve(
        f"https://github.com/godotengine/godot-angle-static/releases/download/{angle_version}/{angle_filename}",
        angle_archive,
    )
    if os.path.exists(angle_folder):
        print(f"Removing existing local ANGLE installation in {angle_folder} ...")
        shutil.rmtree(angle_folder)
    print(f"Extracting ANGLE {angle_filename} to {angle_folder} ...")
    shutil.unpack_archive(angle_archive, angle_folder)
    os.remove(angle_archive)
print("ANGLE installed successfully.\n")

# Complete message
color_print(f'{Ansi.GREEN}All ANGLE components were installed to "{deps_folder}" successfully!')
color_print(f'{Ansi.GREEN}You can now build Godot with statically linked ANGLE by running "scons angle=yes".')

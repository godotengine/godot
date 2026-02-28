#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import argparse
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from misc.utility.color import Ansi, color_print

parser = argparse.ArgumentParser(description="Install GameInput dependencies for Windows platforms.")
parser.add_argument(
    "--mingw_prefix",
    default=os.getenv("MINGW_PREFIX", ""),
    help="Explicitly specify a path containing the MinGW bin folder.",
)
args = parser.parse_args()

# Base Godot dependencies path
# If cross-compiling (no LOCALAPPDATA), we install in `bin`
deps_folder = os.getenv("LOCALAPPDATA")
if deps_folder:
    deps_folder = os.path.join(deps_folder, "Godot", "build_deps")
else:
    deps_folder = os.path.join("bin", "build_deps")

# Check for latest version: https://www.nuget.org/api/v2/package/Microsoft.GameInput (check downloaded filename)
gameinput_version = "3.2.135"
gameinput_archive = os.path.join(deps_folder, f"microsoft.gameinput.{gameinput_version}.nupkg")
gameinput_folder = os.path.join(deps_folder, "gameinput")

# Create dependencies folder
if not os.path.exists(deps_folder):
    os.makedirs(deps_folder)

if os.path.isfile(gameinput_archive):
    os.remove(gameinput_archive)
print(f"Downloading GameInput {gameinput_version} ...")
urllib.request.urlretrieve(
    f"https://www.nuget.org/api/v2/package/Microsoft.GameInput/{gameinput_version}", gameinput_archive
)
if os.path.exists(gameinput_folder):
    print(f"Removing existing local GameInput installation in {gameinput_folder} ...")
    shutil.rmtree(gameinput_folder)
print(f"Extracting GameInput {gameinput_version} to {gameinput_folder} ...")
shutil.unpack_archive(gameinput_archive, gameinput_folder, "zip")
os.remove(gameinput_archive)
print(f"GameInput {gameinput_version} installed successfully.\n")

# Complete message
color_print(f'{Ansi.GREEN}All GameInput components were installed to "{deps_folder}" successfully!')
color_print(f'{Ansi.GREEN}You can now build Godot with GameInput support enabled by running "scons gameinput=yes".')

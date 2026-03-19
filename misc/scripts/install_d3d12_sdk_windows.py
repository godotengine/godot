#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from misc.utility.color import Ansi, color_print

parser = argparse.ArgumentParser(description="Install D3D12 dependencies for Windows platforms.")
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

# Mesa NIR
# Sync with `drivers/d3d12/SCsub` when updating Mesa.
# Check for latest version: https://github.com/godotengine/godot-nir-static/releases/latest
mesa_version = "25.3.1-1"
# WinPixEventRuntime
# Check for latest version: https://www.nuget.org/api/v2/package/WinPixEventRuntime (check downloaded filename)
pix_version = "1.0.240308001"
pix_archive = os.path.join(deps_folder, f"WinPixEventRuntime_{pix_version}.nupkg")
pix_folder = os.path.join(deps_folder, "pix")
# DirectX 12 Agility SDK
# Check for latest version: https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12 (check downloaded filename)
# After updating this, remember to change the default value of the `rendering/rendering_device/d3d12/agility_sdk_version`
# project setting to match the minor version (e.g. for `1.618.5`, it should be `618`).
agility_sdk_version = "1.618.5"
agility_sdk_archive = os.path.join(deps_folder, f"Agility_SDK_{agility_sdk_version}.nupkg")
agility_sdk_folder = os.path.join(deps_folder, "agility_sdk")

# Create dependencies folder
if not os.path.exists(deps_folder):
    os.makedirs(deps_folder)

# Mesa NIR
color_print(f"{Ansi.BOLD}[1/3] Mesa NIR")
for arch in [
    "arm64-llvm",
    "arm64-msvc",
    "x86_32-gcc",
    "x86_32-llvm",
    "x86_32-msvc",
    "x86_64-gcc",
    "x86_64-llvm",
    "x86_64-msvc",
]:
    mesa_filename = "godot-nir-static-" + arch + "-release.zip"
    mesa_archive = os.path.join(deps_folder, mesa_filename)
    mesa_folder = os.path.join(deps_folder, "mesa-" + arch)

    if os.path.isfile(mesa_archive):
        os.remove(mesa_archive)
    print(f"Downloading Mesa NIR {mesa_filename} ...")
    urllib.request.urlretrieve(
        f"https://github.com/godotengine/godot-nir-static/releases/download/{mesa_version}/{mesa_filename}",
        mesa_archive,
    )
    if os.path.exists(mesa_folder):
        print(f"Removing existing local Mesa NIR installation in {mesa_folder} ...")
        shutil.rmtree(mesa_folder)
    print(f"Extracting Mesa NIR {mesa_filename} to {mesa_folder} ...")
    shutil.unpack_archive(mesa_archive, mesa_folder)
    os.remove(mesa_archive)
print("Mesa NIR installed successfully.\n")

# WinPixEventRuntime

# MinGW needs DLLs converted with dlltool.
# We rely on finding gendef/dlltool to detect if we have MinGW.
# Check existence of needed tools for generating mingw library.
pathstr = os.environ.get("PATH", "")
if args.mingw_prefix:
    pathstr = os.path.join(args.mingw_prefix, "bin") + os.pathsep + pathstr
gendef = shutil.which("x86_64-w64-mingw32-gendef", path=pathstr) or shutil.which("gendef", path=pathstr) or ""
dlltool = shutil.which("x86_64-w64-mingw32-dlltool", path=pathstr) or shutil.which("dlltool", path=pathstr) or ""
has_mingw = gendef != "" and dlltool != ""

color_print(f"{Ansi.BOLD}[2/3] WinPixEventRuntime")
if os.path.isfile(pix_archive):
    os.remove(pix_archive)
print(f"Downloading WinPixEventRuntime {pix_version} ...")
urllib.request.urlretrieve(f"https://www.nuget.org/api/v2/package/WinPixEventRuntime/{pix_version}", pix_archive)
if os.path.exists(pix_folder):
    print(f"Removing existing local WinPixEventRuntime installation in {pix_folder} ...")
    shutil.rmtree(pix_folder)
print(f"Extracting WinPixEventRuntime {pix_version} to {pix_folder} ...")
shutil.unpack_archive(pix_archive, pix_folder, "zip")
os.remove(pix_archive)
if has_mingw:
    print("Adapting WinPixEventRuntime to also support MinGW alongside MSVC.")
    cwd = os.getcwd()
    os.chdir(pix_folder)
    subprocess.run([gendef, "./bin/x64/WinPixEventRuntime.dll"])
    subprocess.run(
        [dlltool]
        + "--machine i386:x86-64 --no-leading-underscore -d WinPixEventRuntime.def -D WinPixEventRuntime.dll -l ./bin/x64/libWinPixEventRuntime.a".split()
    )
    subprocess.run([gendef, "./bin/ARM64/WinPixEventRuntime.dll"])
    subprocess.run(
        [dlltool]
        + "--machine arm64 --no-leading-underscore -d WinPixEventRuntime.def -D WinPixEventRuntime.dll -l ./bin/ARM64/libWinPixEventRuntime.a".split()
    )
    os.chdir(cwd)
else:
    print(
        'MinGW support requires "dlltool" and "gendef" dependencies, so only MSVC support is provided for WinPixEventRuntime. Did you forget to provide a `--mingw_prefix`?'
    )
print(f"WinPixEventRuntime {pix_version} installed successfully.\n")

# DirectX 12 Agility SDK
color_print(f"{Ansi.BOLD}[3/3] DirectX 12 Agility SDK")
if os.path.isfile(agility_sdk_archive):
    os.remove(agility_sdk_archive)
print(f"Downloading DirectX 12 Agility SDK {agility_sdk_version} ...")
urllib.request.urlretrieve(
    f"https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/{agility_sdk_version}", agility_sdk_archive
)
if os.path.exists(agility_sdk_folder):
    print(f"Removing existing local DirectX 12 Agility SDK installation in {agility_sdk_folder} ...")
    shutil.rmtree(agility_sdk_folder)
print(f"Extracting DirectX 12 Agility SDK {agility_sdk_version} to {agility_sdk_folder} ...")
shutil.unpack_archive(agility_sdk_archive, agility_sdk_folder, "zip")
os.remove(agility_sdk_archive)
print(f"DirectX 12 Agility SDK {agility_sdk_version} installed successfully.\n")

# Complete message
color_print(f'{Ansi.GREEN}All Direct3D 12 SDK components were installed to "{deps_folder}" successfully!')
color_print(f'{Ansi.GREEN}You can now build Godot with Direct3D 12 support enabled by running "scons d3d12=yes".')

#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))


# Base Godot dependencies path
# If cross-compiling (no LOCALAPPDATA), we install in `bin`
deps_folder = os.getenv("LOCALAPPDATA")
if deps_folder:
    deps_folder = os.path.join(deps_folder, "Godot", "build_deps")
else:
    deps_folder = os.path.join("bin", "build_deps")

# WinRT
winrt_version = "72"

# Create dependencies folder
if not os.path.exists(deps_folder):
    os.makedirs(deps_folder)

winrt_filename = "winrt-headers.zip"
winrt_archive = os.path.join(deps_folder, winrt_filename)
winrt_folder = os.path.join(deps_folder, "winrt_mingw")

if os.path.isfile(winrt_archive):
    os.remove(winrt_archive)

print(f"Downloading WinRT {winrt_filename} ...")
urllib.request.urlretrieve(
    f"https://github.com/godotengine/winrt-mingw/releases/download/{winrt_version}/{winrt_filename}",
    winrt_archive,
)
if os.path.exists(winrt_folder):
    print(f"Removing existing local WinRT installation in {winrt_folder} ...")
    shutil.rmtree(winrt_folder)
print(f"Extracting WinRT {winrt_filename} to {winrt_folder} ...")
shutil.unpack_archive(winrt_archive, winrt_folder)
os.remove(winrt_archive)

print("WinRT installed successfully.\n")

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

# AccessKit
ac_version = "0.21.1"

# Create dependencies folder
if not os.path.exists(deps_folder):
    os.makedirs(deps_folder)

ac_filename = "accesskit-c-" + ac_version + ".zip"
ac_archive = os.path.join(deps_folder, "accesskit.zip")
ac_folder = os.path.join(deps_folder, "accesskit")

if os.path.isfile(ac_archive):
    os.remove(ac_archive)

print(f"Downloading AccessKit {ac_filename} ...")
urllib.request.urlretrieve(
    f"https://github.com/godotengine/godot-accesskit-c-static/releases/download/{ac_version}/{ac_filename}",
    ac_archive,
)
if os.path.exists(ac_folder):
    print(f"Removing existing local AccessKit installation in {ac_folder} ...")
    shutil.rmtree(ac_folder)
print(f"Extracting AccessKit {ac_filename} to {ac_folder} ...")
shutil.unpack_archive(ac_archive, deps_folder)
os.remove(ac_archive)
os.rename(os.path.join(deps_folder, "accesskit-c-" + ac_version), ac_folder)

print("AccessKit installed successfully.\n")

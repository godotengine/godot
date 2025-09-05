#!/usr/bin/env python

import os
import shutil
import sys
import tempfile
import urllib.request
from zipfile import ZipFile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from misc.utility.color import Ansi, color_print

# Swappy
# Check for latest version: https://github.com/godotengine/godot-swappy/releases/latest
swappy_tag = "from-source-2025-01-31"
swappy_filename = "godot-swappy.zip"
swappy_folder = "thirdparty/swappy-frame-pacing"
swappy_archs = [
    "arm64-v8a",
    "armeabi-v7a",
    "x86",
    "x86_64",
]

swappy_archive_destination = os.path.join(tempfile.gettempdir(), swappy_filename)

if os.path.isfile(swappy_archive_destination):
    os.remove(swappy_archive_destination)

print(f"Downloading Swappy {swappy_tag} ...")
urllib.request.urlretrieve(
    f"https://github.com/godotengine/godot-swappy/releases/download/{swappy_tag}/{swappy_filename}",
    swappy_archive_destination,
)

for arch in swappy_archs:
    folder = os.path.join(swappy_folder, arch)
    if os.path.exists(folder):
        print(f"Removing existing local Swappy installation in {folder} ...")
        shutil.rmtree(folder)

print(f"Extracting Swappy {swappy_tag} to {swappy_folder} ...")
with ZipFile(swappy_archive_destination, "r") as zip_file:
    for arch in swappy_archs:
        zip_file.getinfo(f"{arch}/libswappy_static.a").filename = os.path.join(
            swappy_folder, f"{arch}/libswappy_static.a"
        )
        zip_file.extract(f"{arch}/libswappy_static.a")
os.remove(swappy_archive_destination)
print("Swappy installed successfully.\n")

# Complete message
color_print(f'{Ansi.GREEN}Swappy was installed to "{swappy_folder}" successfully!')
color_print(
    f'{Ansi.GREEN}You can now build Godot with Swappy support enabled by running "scons platform=android swappy=yes".'
)

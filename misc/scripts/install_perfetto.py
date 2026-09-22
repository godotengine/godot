#!/usr/bin/env python

import os
import sys
import tempfile
import urllib.request
from zipfile import ZipFile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from misc.utility.color import Ansi, color_print


def get_latest_tag():
    import json

    url = "https://api.github.com/repos/google/perfetto/releases/latest"
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    return data["tag_name"]


# Perfetto
# Check for latest version: https://github.com/google/perfetto/releases/latest
perfetto_tag = get_latest_tag()
perfetto_filename = "perfetto-cpp-sdk-src.zip"
perfetto_folder = "thirdparty/perfetto"

perfetto_archive_destination = os.path.join(tempfile.gettempdir(), perfetto_filename)

if os.path.isfile(perfetto_archive_destination):
    os.remove(perfetto_archive_destination)

print(f"Downloading Perfetto {perfetto_tag} ...")
urllib.request.urlretrieve(
    f"https://github.com/google/perfetto/releases/download/{perfetto_tag}/{perfetto_filename}",
    perfetto_archive_destination,
)

print(f"Extracting Perfetto {perfetto_tag} to {perfetto_folder} ...")
with ZipFile(perfetto_archive_destination, "r") as zip_file:
    zip_file.extractall(perfetto_folder)
os.remove(perfetto_archive_destination)
print("Perfetto installed successfully.\n")

# Complete message
color_print(f'{Ansi.GREEN}Perfetto was installed to "{perfetto_folder}" successfully!')

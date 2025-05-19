#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys

if len(sys.argv) < 2:
    print("Invalid usage of dotnet_format.py, it should be called with a path to one or multiple files.")
    sys.exit(1)

# Create dummy generated files, if needed.
for path in [
    "modules/mono/SdkPackageVersions.props",
]:
    if os.path.exists(path):
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("<Project />")

# Avoid importing GeneratedIncludes.props.
os.environ["GodotSkipGenerated"] = "true"

# Match all the input files to their respective C# project.
projects = {
    path: " ".join([f for f in sys.argv[1:] if os.path.commonpath([f, path]) == path])
    for path in [os.path.dirname(f) for f in glob.glob("**/*.csproj", recursive=True)]
}

# Run dotnet format on all projects with more than 0 modified files.
for path, files in projects.items():
    if files:
        os.system(f"dotnet format {path} --include {files}")

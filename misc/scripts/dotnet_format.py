#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys

# Create dummy generated files.
for path in [
    "modules/mono/SdkPackageVersions.props",
]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("<Project />")

# Avoid importing GeneratedIncludes.props.
os.environ["GodotSkipGenerated"] = "true"

# Match all the input files to their respective C# project.
input_files = [os.path.normpath(x) for x in sys.argv]
projects = {
    path: [f for f in sys.argv if os.path.commonpath([f, path]) == path]
    for path in [os.path.dirname(f) for f in glob.glob("**/*.csproj", recursive=True)]
}

# Run dotnet format on all projects with more than 0 modified files.
for path, files in projects.items():
    if len(files) > 0:
        command = f"dotnet format {path} --include {' '.join(files)}"
        os.system(command)

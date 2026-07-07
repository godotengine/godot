#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Invalid usage of header_guards.py, it should be called with a path to one or multiple files.")
    sys.exit(1)

changed = []
invalid = []

for file in sys.argv[1:]:
    header_start = -1
    header_end = -1

    with open(file.strip(), "rt", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        sline = line.strip()

        if header_start < 0:
            if sline == "":  # Skip empty lines at the top.
                continue

            if sline.startswith("/**********"):  # Godot header starts this way.
                header_start = idx
            else:
                header_end = 0  # There is no Godot header.
                break
        else:
            if not sline.startswith(("*", "/*")):  # Not in the Godot header anymore.
                header_end = idx + 1  # The guard should be two lines below the Godot header.
                break

    if (HEADER_CHECK_OFFSET := header_end) < 0 or HEADER_CHECK_OFFSET >= len(lines):
        invalid.append(file)
        continue

    if lines[HEADER_CHECK_OFFSET].startswith("#pragma once"):
        continue

    # Might be using legacy header guards.
    HEADER_BEGIN_OFFSET = HEADER_CHECK_OFFSET + 1
    HEADER_END_OFFSET = len(lines) - 1

    if HEADER_BEGIN_OFFSET >= HEADER_END_OFFSET:
        invalid.append(file)
        continue

    if (
        lines[HEADER_CHECK_OFFSET].startswith("#ifndef")
        and lines[HEADER_BEGIN_OFFSET].startswith("#define")
        and lines[HEADER_END_OFFSET].startswith("#endif")
    ):
        lines[HEADER_CHECK_OFFSET] = "#pragma once"
        lines[HEADER_BEGIN_OFFSET] = "\n"
        lines.pop()
        with open(file, "wt", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        changed.append(file)
        continue

    # Verify `#pragma once` doesn't exist at invalid location.
    misplaced = False
    for line in lines:
        if line.startswith("#pragma once"):
            misplaced = True
            break

    if misplaced:
        invalid.append(file)
        continue

    # Assume that we're simply missing a guard entirely.
    lines.insert(HEADER_CHECK_OFFSET, "#pragma once\n\n")
    with open(file, "wt", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    changed.append(file)

if changed:
    for file in changed:
        print(f"FIXED: {file}")
if invalid:
    for file in invalid:
        print(f"REQUIRES MANUAL CHANGES: {file}")
    sys.exit(1)

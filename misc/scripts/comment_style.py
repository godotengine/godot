#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

if len(sys.argv) < 2:
    print("Invalid usage of comment_style.py, it should be called with a path to one or multiple files.")
    sys.exit(1)

IS_COMMENT = re.compile(r"// .*?\w")
CARET_COMMENT = re.compile(r"// *\^")
PLUS_COMMENT = re.compile(r"// *\+")
COMMENTS_TO_SKIP = [CARET_COMMENT, PLUS_COMMENT]

fixes = []

for file in sys.argv[1:]:
    with open(file.strip(), "rt", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        sline = line.strip()
        m = IS_COMMENT.match(sline)
        if m is not None:
            # Check if we should skip this comment
            if any(regex.match(sline) is not None for regex in COMMENTS_TO_SKIP):
                continue

            # If the next line is also a comment it's hard to know if a `.` is needed, so it's better to skip it
            # and let a reviewer catch it
            if (idx + 1) < len(lines) and IS_COMMENT.match(lines[idx + 1]) is not None:
                continue

            # It's a comment, check if ends with any of this characters
            if not sline.endswith((".", ":", ",", ";")):
                # Files are not 0 indexed
                fixes.append((file, idx + 1, sline, sline + "."))

if fixes:
    for fix in fixes:
        file, idx, line_with_error, line_fixed = fix
        print(f"REQUIRES MANUAL CHANGES: {file}:{idx}")
        print(f"- {line_with_error}")
        print(f"+ {line_fixed}")
    sys.exit(1)

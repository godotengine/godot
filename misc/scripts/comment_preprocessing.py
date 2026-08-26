#!/usr/bin/env python3

if __name__ != "__main__":
    raise ImportError(f'Utility script "{__file__}" should not be used as a module!')

import os
from pathlib import Path

if Path(os.getcwd()).as_posix() != (ROOT := Path(__file__).parent.parent.parent).as_posix():
    raise RuntimeError(f'Utility script "{__file__}" must be run from the repository root!')

import argparse
import re

SKIP_DIRS = {"thirdparty"}
DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")

MIN_BODY_LINES = 4


def process(path):
    with open(path, encoding="utf-8", newline="") as file:
        lines = file.readlines()

    stack = []
    changed = False

    for i, line in enumerate(lines):
        match = DIRECTIVE.match(line)
        if not match:
            continue

        directive, rest = match.groups()
        condition = re.sub(r"/\*.*?\*/|//.*", "", rest).strip()

        if directive in ("ifdef", "ifndef", "if"):
            stack.append({
                "condition": condition,
                "has_elif": False,
                "body_start": i + 1,
            })
        elif directive == "elif":
            if stack:
                stack[-1]["condition"] = condition
                stack[-1]["has_elif"] = True
                stack[-1]["body_start"] = i + 1
        elif directive == "else":
            if stack and "//" not in rest and "/*" not in rest:
                body_lines = i - stack[-1]["body_start"]

                if body_lines >= MIN_BODY_LINES:
                    lines[i] = line.rstrip("\r\n") + f" // {stack[-1]['condition']}\n"

                    if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("//"):
                        lines[i + 1] = "\n" + lines[i + 1]

                    changed = True

                stack[-1]["body_start"] = i + 1
        elif directive == "endif":
            if stack:
                entry = stack.pop()
                body_lines = i - entry["body_start"]

                if entry["has_elif"] or body_lines < MIN_BODY_LINES:
                    continue

                if "//" not in rest and "/*" not in rest:
                    lines[i] = line.rstrip("\r\n") + f" // {entry['condition']}\n"

                    if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("//"):
                        lines[i + 1] = "\n" + lines[i + 1]
                    changed = True

    if not changed:
        return changed
    with open(path, "w", encoding="utf-8", newline="") as file:
        file.writelines(lines)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Check for relevant comments for preprocessor derictives.")
    parser.add_argument("files", nargs="+", help="A list of files to validate")
    args = parser.parse_args()

    changed = []
    for file in map(Path, args.files):
        try:
            if process(file):
                changed.append(file)
        except (OSError, UnicodeDecodeError) as e:
            print(f"Skipping: {file} ({e})")

    for file in changed:
        print(f"Comment(s) added: {file}")

    return 0


if __name__ == "__main__":
    main()

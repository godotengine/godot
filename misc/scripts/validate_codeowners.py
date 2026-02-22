#!/usr/bin/env python3

if __name__ != "__main__":
    raise SystemExit(f'Utility script "{__file__}" should not be used as a module!')

import argparse
import re
import subprocess
import sys

sys.path.insert(0, "./")

try:
    from methods import print_error, print_info
except ImportError:
    raise SystemExit(f"Utility script {__file__} must be run from repository root!")


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """Convert a CODEOWNERS glob to a RegEx pattern."""

    # Heavily inspired by: https://github.com/hmarr/codeowners/blob/main/match.go

    # Handle specific edgecases first.
    if "***" in glob:
        raise SyntaxError("Pattern cannot contain three consecutive asterisks")
    if glob == "/":
        raise SyntaxError('Standalone "/" will not match anything')
    if not glob:
        raise ValueError("Empty pattern")

    segments = glob.split("/")
    if not segments[0]:
        # Leading slash; relative to root.
        segments = segments[1:]
    else:
        # Check for single-segment pattern, which matches relative to any descendent path.
        #  This is equivalent to a leading `**/`.
        if len(segments) == 1 or (len(segments) == 2 and not segments[1]):
            if segments[0] != "**":
                segments.insert(0, "**")

    if len(segments) > 1 and not segments[-1]:
        # A trailing slash is equivalent to `/**`.
        segments[-1] = "**"

    last_index = len(segments) - 1
    need_slash = False
    pattern = r"\A"

    for index, segment in enumerate(segments):
        if segment == "**":
            if index == 0 and index == last_index:
                pattern += r".+"  # Pattern is just `**`; match everything.
            elif index == 0:
                pattern += r"(?:.+/)?"  # Pattern starts with `**`; match any leading path segment.
                need_slash = False
            elif index == last_index:
                pattern += r"/.*"  # Pattern ends with `**`; match any trailing path segment.
            else:
                pattern += r"(?:/.+)?"  # Pattern contains `**`; match zero or more path segments.
                need_slash = True

        elif segment == "*":
            if need_slash:
                pattern += "/"
            # Regular wildcard; match any non-separator characters.
            pattern += r"[^/]+"
            need_slash = True

        else:
            if need_slash:
                pattern += "/"

            escape = False
            for char in segment:
                if escape:
                    escape = False
                    pattern += re.escape(char)
                    continue
                elif char == "\\":
                    escape = True
                elif char == "*":
                    # Multi-character wildcard.
                    pattern += r"[^/]*"
                elif char == "?":
                    # Single-character wildcard.
                    pattern += r"[^/]"
                else:
                    # Regular character
                    pattern += re.escape(char)

            if index == last_index:
                pattern += r"(?:/.*)?"  # No trailing slash; match descendent paths.
            need_slash = True

    pattern += r"\Z"
    return re.compile(pattern)


RE_CODEOWNERS = re.compile(r"^(?P<code>[^#](?:\\ |[^\s])+) +(?P<owners>(?:[^#][^\s]+ ?)+)")


def parse_codeowners() -> list[tuple[re.Pattern[str], list[str]]]:
    codeowners = []
    with open(".github/CODEOWNERS", encoding="utf-8", newline="\n") as file:
        for line in reversed(file.readlines()):  # Lower items have higher precedence.
            if match := RE_CODEOWNERS.match(line):
                codeowners.append((glob_to_regex(match["code"]), match["owners"].split()))
    return codeowners


def main() -> int:
    parser = argparse.ArgumentParser(description="Utility script for validating CODEOWNERS assignment.")
    parser.add_argument("files", nargs="*", help="A list of files to validate. If excluded, checks all owned files.")
    parser.add_argument("-u", "--unowned", action="store_true", help="Only output files without an owner.")
    args = parser.parse_args()

    files: list[str] = args.files
    if not files:
        files = subprocess.run(["git", "ls-files"], text=True, capture_output=True).stdout.splitlines()

    ret = 0
    codeowners = parse_codeowners()

    for file in files:
        matched = False
        for code, owners in codeowners:
            if code.match(file):
                matched = True
                if not args.unowned:
                    print_info(f"{file}: {owners}")
                break
        if not matched:
            print_error(f"{file}: <UNOWNED>")
            ret += 1

    return ret


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)

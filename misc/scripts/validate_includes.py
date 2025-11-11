if __name__ != "__main__":
    raise ImportError(f"{__name__} should not be used as a module.")

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Assumed to run from repository root
BASE_FOLDER = Path(__file__).resolve().parent.parent.parent
RE_INCLUDES = re.compile(
    r"^[^\S\n]*"  # Begin matching at newline, but allow non-newline whitespace beforehand.
    r"#\s*(?P<keyword>include|import)\s*"  # Both `include` and `import` keywords valid.
    r'(?P<sequence>[<"])'  # Handle both styles of char wrappers. Does NOT handle macro expansion.
    r'(?P<path>.*)[>"]'  # Resolve path of include itself. Can safely assume the sequence matches.
    r"(?:.*/[/\*] validator: "  # `validator: ` serves as our comment pragma declaration.
    r"(?P<instructions>\S+)"  # Assume instructions will be one-word.
    r")?",  # Pragmas are not required, can be safely excluded.
    re.MULTILINE,
)


@dataclass
class IncludeData:
    path: str
    instructions: str
    is_angle: bool
    is_import: bool

    @staticmethod
    def from_match(match: re.Match[str]):
        items = match.groupdict()
        return IncludeData(
            items["path"],
            items["instructions"] or "",
            items["sequence"] == "<",
            items["keyword"] == "import",
        )


def is_relative_to(base: Path, other: Path):
    if sys.version_info > (3, 9):
        return base.is_relative_to(other)
    else:
        try:
            base.relative_to(other)
        except ValueError:
            return False
        return True


def get_encapsulation_level(path: Path):
    # Necessitates earlier check.
    if "tests" in path.parts:
        return 8

    resolved = path.resolve()
    if not is_relative_to(resolved, BASE_FOLDER) or is_relative_to(resolved, BASE_FOLDER / "thirdparty"):
        return 0
    elif is_relative_to(resolved, BASE_FOLDER / "core"):
        return 1
    elif is_relative_to(resolved, BASE_FOLDER / "servers"):
        return 2
    elif is_relative_to(resolved, BASE_FOLDER / "scene"):
        return 3
    elif is_relative_to(resolved, BASE_FOLDER / "editor"):
        return 4
    elif is_relative_to(resolved, BASE_FOLDER / "drivers"):
        return 5
    elif is_relative_to(resolved, BASE_FOLDER / "platform") and resolved.parent == BASE_FOLDER / "platform":
        return 6
    elif is_relative_to(resolved, BASE_FOLDER / "modules"):
        return 7
    elif is_relative_to(resolved, BASE_FOLDER / "tests"):
        return 8
    elif is_relative_to(resolved, BASE_FOLDER / "main"):
        return 9
    else:
        return 10


def parse_file(path: Path) -> int:
    ret = 0
    level = get_encapsulation_level(path)

    if level == 0:
        return ret

    with open(path) as file:
        matches = RE_INCLUDES.finditer(file.read())

    for data in map(IncludeData.from_match, matches):
        if "\\" in data.path:
            print(f'"{path.as_posix()}" contains Windows-style path separators in include: "{data.path}"!')
            ret += 1
            continue

        if data.is_angle:
            # Angle-bracket resolution is much trickier; skip for now.
            continue

        include = path.parent / data.path
        if include.exists():
            # FIXME: Relative includes are currently outside the scope of this validator.
            continue
        else:
            include = BASE_FOLDER / data.path
            if not include.exists():
                # Assume special case (eg: `platform.h`).
                continue

        if data.instructions != "ignore" and level < get_encapsulation_level(include):
            print(f'"{path.as_posix()}" violates encapsulation, should not include "{data.path}"!')
            ret += 1

    return ret


def main():
    parser = argparse.ArgumentParser(description="Validate `#include` encapsulation in the provided C/C++ files")
    parser.add_argument("paths", nargs="*", help="Paths of files to validate")
    args = parser.parse_args()

    ret = 0

    for path in args.paths:
        ret += parse_file(Path(path))

    return ret


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)

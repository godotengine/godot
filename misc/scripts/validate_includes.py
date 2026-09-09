#!/usr/bin/env python3

if __name__ != "__main__":
    raise ImportError(f'Utility script "{__file__}" should not be used as a module!')

import os
from pathlib import Path

if Path(os.getcwd()).as_posix() != (ROOT := Path(__file__).parent.parent.parent).as_posix():
    raise RuntimeError(f'Utility script "{__file__}" must be run from the repository root!')

import argparse
import re
from dataclasses import dataclass

BASE_FOLDER = Path(__file__).resolve().parent.parent.parent
RE_INCLUDES = re.compile(
    r"^#(?P<keyword>include|import) "  # Both `include` and `import` keywords valid.
    r'(?P<sequence>[<"])'  # Handle both styles of char wrappers. Does NOT handle macro expansion.
    r'(?P<path>.+?)[>"]',  # Resolve path of include itself. Can safely assume the sequence matches.
    re.RegexFlag.MULTILINE,
)


@dataclass
class IncludeData:
    path: str
    is_angle: bool
    is_import: bool

    @staticmethod
    def from_match(match: re.Match[str]):
        items = match.groupdict()
        return IncludeData(
            items["path"],
            items["sequence"] == "<",
            items["keyword"] == "import",
        )

    def copy(self):
        return IncludeData(self.path, self.is_angle, self.is_import)

    def __str__(self):
        return "#{keyword} {rbracket}{path}{lbracket}".format(
            keyword="import" if self.is_import else "include",
            rbracket="<" if self.is_angle else '"',
            path=self.path,
            lbracket=">" if self.is_angle else '"',
        )


def validate_includes(path: Path) -> int:
    ret = 0
    content = path.read_text(encoding="utf-8")

    for data in map(IncludeData.from_match, RE_INCLUDES.finditer(content)):
        original_data = data.copy()

        if "\\" in data.path:
            data.path = data.path.replace("\\", "/")

        if data.path.startswith("thirdparty/"):
            data.is_angle = True

        if (relative_path := path.parent / data.path).exists():
            # Relative includes are only permitted under certain circumstances.

            if relative_path.name.split(".")[0] == path.name.split(".")[0]:
                # Identical leading names permitted
                pass

            elif ("modules" in relative_path.parts and "modules" in path.parts) or (
                "platform" in relative_path.parts and "platform" in path.parts
            ):
                # Modules and platforms can use relative includes if constrained to the module/platform itself.
                pass

            else:
                data.path = relative_path.resolve().relative_to(BASE_FOLDER).as_posix()

        if original_data != data:
            content = content.replace(f"{original_data}", f"{data}")
            ret += 1

    if ret:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)

    return ret


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate C/C++ includes, correcting if necessary")
    parser.add_argument("files", nargs="+", help="A list of files to validate")
    args = parser.parse_args()

    ret = 0

    for file in map(Path, args.files):
        ret += validate_includes(file)

    return ret


try:
    raise SystemExit(main())
except KeyboardInterrupt:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)

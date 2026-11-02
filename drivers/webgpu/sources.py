"""Shared source selection for SCons and the standalone Tint compiler."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _files(dependency, suffix):
    stamp = ROOT / "thirdparty" / dependency / ".webgpu-source.json"
    return sorted(file for file in json.loads(stamp.read_text(encoding="utf-8"))["files"] if file.endswith(suffix))


def _platform_sources(platform):
    command, temporary, environment, executable, terminal, text = {
        "windows": ("windows", "windows", "windows", "executable_path_windows", "windows", "windows"),
        "macos": ("posix", "posix", "other", "executable_file_mac", "posix", "posix"),
        "linuxbsd": ("posix", "posix", "other", "executable_path_linux", "posix", "posix"),
    }.get(platform, ("other", "other", "other", None, "other", "other"))
    files = [
        f"command/command_{command}",
        f"file/tmpfile_{temporary}",
        f"system/env_{environment}",
        f"system/terminal_{terminal}",
        f"text/styled_text_printer_{text}",
    ]
    if executable:
        files.append(f"system/{executable}")
    if platform in ("linuxbsd", "macos"):
        files.append("text/styled_text_printer_ansi")
    return {f"src/tint/utils/{file}.cc" for file in files}


def tint_sources(platform):
    platform_files = set().union(*(_platform_sources(p) for p in ("windows", "macos", "linuxbsd", "web")))
    disabled = {"src/tint/lang/core/ir/binary/decode.cc", "src/tint/lang/core/ir/binary/encode.cc"}
    return sorted((set(_files("tint", ".cc")) - platform_files - disabled) | _platform_sources(platform))


def spirv_tools_sources():
    return [file for file in _files("spirv-tools", ".cpp") if file != "source/opt/pch_source_opt.cpp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dependency", choices=("tint", "spirv-tools"))
    parser.add_argument("--platform", default={"win32": "windows", "darwin": "macos"}.get(sys.platform, "linuxbsd"))
    args = parser.parse_args()
    files = tint_sources(args.platform) if args.dependency == "tint" else spirv_tools_sources()
    print("\n".join(f"thirdparty/{args.dependency}/{file}" for file in files))

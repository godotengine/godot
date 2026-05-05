"""Functions used to generate source files during build time"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import methods
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

import methods  # noqa E402


def export_icon_builder(target, source):
    src_path = Path(str(source[0]))
    src_name = src_path.stem
    platform = src_path.parent.parent.stem

    with open(str(source[0]), "r", encoding="utf-8") as file:
        svg = file.read()

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
inline constexpr const char *_{platform}_{src_name}_svg = {methods.to_raw_cstring(svg)};
"""
        )


def register_platform_apis_builder(target, source, env):
    platforms = source[0].read()
    api_inc = "\n".join([f'#include "{p}/api/api.h"' for p in platforms])
    api_reg = "\n\t".join([f"register_{p}_api();" for p in platforms])
    api_unreg = "\n\t".join([f"unregister_{p}_api();" for p in platforms])
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
#include "register_platform_apis.h"

{api_inc}

void register_platform_apis() {{
	{api_reg}
}}

void unregister_platform_apis() {{
	{api_unreg}
}}
"""
        )


def main():
    parser = argparse.ArgumentParser(description="Platform build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["export_icon_builder"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "export_icon_builder":
        export_icon_builder(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

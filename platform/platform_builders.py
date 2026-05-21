"""Functions used to generate source files during build time"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import methods


def export_icon_builder(target, source):
    src_path = Path(str(source[0]))
    src_name = src_path.stem
    platform = src_path.parent.parent.stem

    with open(str(source[0]), "r", encoding="utf-8") as file:
        svg = file.read()

    with methods.generated_wrapper(target) as file:
        file.write(
            f"""\
inline constexpr const char *_{platform}_{src_name}_svg = {methods.to_raw_cstring(svg)};
"""
        )


def register_platform_apis_builder(target, source):
    platforms = source
    api_inc = "\n".join([f'#include "{p}/api/api.h"' for p in platforms])
    api_reg = "\n\t".join([f"register_{p}_api();" for p in platforms])
    api_unreg = "\n\t".join([f"unregister_{p}_api();" for p in platforms])
    with methods.generated_wrapper(target) as file:
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
    # Parse initial arguments to check for argfile
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--argfile", help="File containing additional arguments")
    initial_args, remaining_args = initial_parser.parse_known_args()

    # If argfile is provided, read arguments from it
    if initial_args.argfile:
        file_args = methods.read_args_from_file(initial_args.argfile)
        # Combine file arguments with remaining command line arguments
        sys.argv = [sys.argv[0]] + file_args + remaining_args

        # Print arguments to stdout if --verbose is present
        if "--verbose" in sys.argv:
            print("Arguments read from file:", initial_args.argfile)
            print("Combined arguments:", " ".join(file_args + remaining_args))

    # Parse all arguments
    parser = argparse.ArgumentParser(description="Platform build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["export_icon_builder", "register_platform_apis_builder"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", required=True, help="Target file")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "export_icon_builder":
        export_icon_builder(target, source)
    elif args.method == "register_platform_apis_builder":
        register_platform_apis_builder(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

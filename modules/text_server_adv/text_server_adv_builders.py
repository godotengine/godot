"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import methods


def make_icu_data(target, source):
    buffer = methods.get_buffer(str(source[0]))
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
/* (C) 2016 and later: Unicode, Inc. and others. */
/* License & terms of use: https://www.unicode.org/copyright.html */

#include <unicode/utypes.h>
#include <unicode/udata.h>
#include <unicode/uversion.h>

extern "C" U_EXPORT const size_t U_ICUDATA_SIZE = {len(buffer)};
extern "C" U_EXPORT const unsigned char U_ICUDATA_ENTRY_POINT[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def main():
    parser = argparse.ArgumentParser(description="Text server advanced build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_icu_data"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_icu_data":
        make_icu_data(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

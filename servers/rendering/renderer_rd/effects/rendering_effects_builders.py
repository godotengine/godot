"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../"))

import methods


def areatex_builder(target, source):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#define AREATEX_WIDTH 160
#define AREATEX_HEIGHT 560
#define AREATEX_PITCH (AREATEX_WIDTH * 2)
#define AREATEX_SIZE (AREATEX_HEIGHT * AREATEX_PITCH)

inline constexpr const unsigned char area_tex_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def searchtex_builder(target, source):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#define SEARCHTEX_WIDTH 64
#define SEARCHTEX_HEIGHT 16
#define SEARCHTEX_PITCH SEARCHTEX_WIDTH
#define SEARCHTEX_SIZE (SEARCHTEX_HEIGHT * SEARCHTEX_PITCH)

inline constexpr const unsigned char search_tex_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def main():
    parser = argparse.ArgumentParser(description="Rendering effects build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["areatex_builder", "searchtex_builder"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "areatex_builder":
        areatex_builder(target, source)
    elif args.method == "searchtex_builder":
        searchtex_builder(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

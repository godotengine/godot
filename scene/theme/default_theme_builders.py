"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "..", ".."))

import methods  # noqa E402


def make_fonts_header(target, source):
    with methods.generated_wrapper(str(target[0])) as file:
        for src in map(str, source):
            # Saving uncompressed, since FreeType will reference from memory pointer.
            buffer = methods.get_buffer(src)
            name = os.path.splitext(os.path.basename(src))[0]

            file.write(f"""\
inline constexpr int _font_{name}_size = {len(buffer)};
inline constexpr unsigned char _font_{name}[] = {{
{methods.format_buffer(buffer, 1)}
}};

""")


def main():
    parser = argparse.ArgumentParser(description="Default theme build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_fonts_header"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_fonts_header":
        make_fonts_header(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

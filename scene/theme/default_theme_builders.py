"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import methods


def make_fonts_header(target, source):
    with methods.generated_wrapper(target) as file:
        # Saving uncompressed, since FreeType will reference from memory pointer.
        buffer = methods.get_buffer(source)
        name = os.path.splitext(os.path.basename(source))[0]

        file.write(f"""\
inline constexpr int _font_{name}_size = {len(buffer)};
inline constexpr unsigned char _font_{name}[] = {{
{methods.format_buffer(buffer, 1)}
}};

""")


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
    parser = argparse.ArgumentParser(description="Default theme build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_fonts_header"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", required=True, help="Target file")
    parser.add_argument("--source", required=True, help="Source file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

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

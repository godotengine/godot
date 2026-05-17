"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

import methods


# See also `editor/icons/editor_icons_builders.py`.
def make_default_theme_icons_action(target, source):
    icons_names = []
    icons_raw = []

    for src in map(str, source):
        with open(src, encoding="utf-8", newline="\n") as file:
            icons_raw.append(methods.to_raw_cstring(file.read()))

        name = os.path.splitext(os.path.basename(src))[0]
        icons_names.append(f'"{name}"')

    icons_names_str = ",\n\t".join(icons_names)
    icons_raw_str = ",\n\t".join(icons_raw)

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
inline constexpr int default_theme_icons_count = {len(icons_names)};
inline constexpr const char *default_theme_icons_sources[] = {{
	{icons_raw_str}
}};

inline constexpr const char *default_theme_icons_names[] = {{
	{icons_names_str}
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
    parser = argparse.ArgumentParser(description="Default theme icons build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_default_theme_icons_action"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_default_theme_icons_action":
        make_default_theme_icons_action(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import methods


# See also `scene/theme/icons/default_theme_icons_builders.py`.
def make_editor_icons_action(target, source):
    icons_names = []
    icons_raw = []
    icons_med = []
    icons_big = []

    for idx, svg in enumerate(source):
        path = str(svg)
        with open(path, encoding="utf-8", newline="\n") as file:
            icons_raw.append(methods.to_raw_cstring(file.read()))

        name = os.path.splitext(os.path.basename(path))[0]
        icons_names.append(f'"{name}"')

        if name.endswith("MediumThumb"):
            icons_med.append(str(idx))
        elif name.endswith(("BigThumb", "GodotFile")):
            icons_big.append(str(idx))

    icons_names_str = ",\n\t".join(icons_names)
    icons_raw_str = ",\n\t".join(icons_raw)

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
inline constexpr int editor_icons_count = {len(icons_names)};
inline constexpr const char *editor_icons_sources[] = {{
	{icons_raw_str}
}};

inline constexpr const char *editor_icons_names[] = {{
	{icons_names_str}
}};

inline constexpr int editor_md_thumbs_count = {len(icons_med)};
inline constexpr int editor_md_thumbs_indices[] = {{ {", ".join(icons_med)} }};

inline constexpr int editor_bg_thumbs_count = {len(icons_big)};
inline constexpr int editor_bg_thumbs_indices[] = {{ {", ".join(icons_big)} }};
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

        # Print arguments to stdout unless --quiet is present
        if "--quiet" not in sys.argv:
            print("Arguments read from file:", initial_args.argfile)
            print("Combined arguments:", " ".join(file_args + remaining_args))

    # Parse all arguments
    parser = argparse.ArgumentParser(description="Editor icons build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_editor_icons_action"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")
    parser.add_argument("--quiet", action="store_true", help="Suppress argument printing")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_editor_icons_action":
        make_editor_icons_action(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

import methods


def make_splash(target, source):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        # Use a neutral gray color to better fit various kinds of projects.
        file.write(f"""\
#include "core/math/color.h"

static const Color boot_splash_bg_color = Color(0.14, 0.14, 0.14);
inline constexpr const unsigned char boot_splash_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def make_splash_editor(target, source):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        # The editor splash background color is taken from the default editor theme's background color.
        # This helps achieve a visually "smoother" transition between the splash screen and the editor.
        file.write(f"""\
#include "core/math/color.h"

static const Color boot_splash_editor_bg_color = Color(0.125, 0.145, 0.192);
inline constexpr const unsigned char boot_splash_editor_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def make_app_icon(target, source):
    buffer = methods.get_buffer(str(source[0]))

    with methods.generated_wrapper(str(target[0])) as file:
        # Use a neutral gray color to better fit various kinds of projects.
        file.write(f"""\
inline constexpr const unsigned char app_icon_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def main():
    parser = argparse.ArgumentParser(description="Main build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["make_splash", "make_splash_editor", "make_app_icon"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", nargs="+", required=True, help="Target file(s)")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_splash":
        make_splash(target, source)
    elif args.method == "make_splash_editor":
        make_splash_editor(target, source)
    elif args.method == "make_app_icon":
        make_app_icon(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Functions used to generate source files during build time"""

import argparse
import sys

try:
    sys.path.insert(0, "./")
    import methods
except ImportError:
    raise SystemExit(f'Generator script "{__file__}" must be run from repository root!')


def splash(target: str, image: str) -> None:
    buffer = methods.get_buffer(image)

    with methods.generated_wrapper(target) as file:
        # Use a neutral gray color to better fit various kinds of projects.
        file.write(f"""\
#include "core/math/color.h"

static const Color boot_splash_bg_color = Color(0.14, 0.14, 0.14);
inline constexpr const unsigned char boot_splash_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def splash_editor(target: str, image: str) -> None:
    buffer = methods.get_buffer(image)

    with methods.generated_wrapper(target) as file:
        # The editor splash background color is taken from the default editor theme's background color.
        # This helps achieve a visually "smoother" transition between the splash screen and the editor.
        file.write(f"""\
#include "core/math/color.h"

static const Color boot_splash_editor_bg_color = Color(0.125, 0.145, 0.192);
inline constexpr const unsigned char boot_splash_editor_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def app_icon(target: str, image: str) -> None:
    buffer = methods.get_buffer(image)

    with methods.generated_wrapper(target) as file:
        # Use a neutral gray color to better fit various kinds of projects.
        file.write(f"""\
inline constexpr const unsigned char app_icon_png[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    splash_parser = subparsers.add_parser("splash")
    splash_parser.add_argument("target")
    splash_parser.add_argument("image")

    splash_editor_parser = subparsers.add_parser("splash_editor")
    splash_editor_parser.add_argument("target")
    splash_editor_parser.add_argument("image")

    app_icon_parser = subparsers.add_parser("app_icon")
    app_icon_parser.add_argument("target")
    app_icon_parser.add_argument("image")

    args = vars(parser.parse_args())
    command = globals().get(args.pop("command"), {})
    command(**args)


if __name__ == "__main__":
    main()

"""Functions used to generate source files during build time"""

import argparse
import os
import sys

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import methods


def modules_enabled_builder(target, source):
    modules = sorted(source)
    with methods.generated_wrapper(target) as file:
        for module in modules:
            file.write(f"#define MODULE_{module.upper()}_ENABLED\n")


def register_module_types_builder(target, source, env):
    modules = source[0].read()
    mod_inc = "\n".join([f'#include "{value}/register_types.h"' for value in modules.values()])
    mod_init = "\n".join([
        f"""\
#ifdef MODULE_{key.upper()}_ENABLED
	initialize_{key}_module(p_level);
#endif"""
        for key in modules.keys()
    ])
    mod_uninit = "\n".join([
        f"""\
#ifdef MODULE_{key.upper()}_ENABLED
	uninitialize_{key}_module(p_level);
#endif"""
        for key in modules.keys()
    ])
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
#include "register_module_types.h"

#include "modules/modules_enabled.gen.h"

// IWYU pragma: begin_keep.
{mod_inc}
// IWYU pragma: end_keep.

void initialize_modules(ModuleInitializationLevel p_level) {{
{mod_init}
}}

void uninitialize_modules(ModuleInitializationLevel p_level) {{
{mod_uninit}
}}
"""
        )


def modules_tests_builder(target, source):
    headers = sorted([os.path.relpath(src, methods.base_folder).replace("\\", "/") for src in source])
    with methods.generated_wrapper(target) as file:
        file.write("// IWYU pragma: begin_keep.\n")
        for header in headers:
            file.write(f'#include "{header}"\n')
        file.write("// IWYU pragma: end_keep.\n")


def main():
    parser = argparse.ArgumentParser(description="Modules build tools")
    parser.add_argument(
        "--method",
        required=True,
        choices=["modules_tests_builder", "modules_enabled_builder"],
        help="Builder method to execute",
    )
    parser.add_argument("--target", required=True, help="Target file")
    parser.add_argument("--source", nargs="*", default=[], help="Source file(s)")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "modules_tests_builder":
        modules_tests_builder(target, source)
    elif args.method == "modules_enabled_builder":
        modules_enabled_builder(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

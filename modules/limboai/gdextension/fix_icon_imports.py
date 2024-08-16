#!/usr/bin/python
"""
Usage: fix_icon_imports.py [--silent] PROJECT_DIR
Fix icon imports in PROJECT_DIR/addons/limboai/icons/.

Options:
  -s, --silent      Don't print anything.
  -h, --help        Print this message.

Dependencies: python3.

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
"""

import os
import glob
import sys
import getopt


def usage():
    print(__doc__.strip())


def get_limboai_icon_import_files(project_path="demo/"):
    if not os.path.isdir(project_path):
        raise FileNotFoundError("Project directory not found: " + project_path)

    icons_path = os.path.join(project_path, "addons/limboai/icons/")
    if not os.path.isdir(icons_path):
        raise FileNotFoundError("Icons directory not found: " + icons_path)

    return glob.glob(icons_path + "*.import")


def fix_import_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found: " + file_path)

    old_lines = []
    new_lines = []

    file = open(file_path, "r")
    old_lines = file.readlines()
    file.close()

    for line in old_lines:
        line = line.replace("editor/scale_with_editor_scale=false", "editor/scale_with_editor_scale=true")
        line = line.replace(
            "editor/convert_colors_with_editor_theme=false", "editor/convert_colors_with_editor_theme=true"
        )
        new_lines.append(line)

    if old_lines != new_lines:
        file = open(file_path, "w")
        for line in new_lines:
            file.write(line)
        file.close()
        return True
    return False


def fix_icon_imports(project_path="demo/", silent=False):
    if not silent:
        print("Checking icon import files...")

    project_import_files = get_limboai_icon_import_files(project_path)

    for import_file in project_import_files:
        changed = fix_import_file(import_file)
        if changed and not silent:
            print("Updated icon import file: " + import_file)


if __name__ == "__main__":
    silent = False
    project_path = "demo/"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "s", ["silent"])
    except getopt.GetoptError as e:
        print(
            "%s: %s!\n"
            % (
                os.path.basename(__file__),
                e.msg,
            )
        )
        usage()
        sys.exit(2)

    if len(args) > 1:
        usage()
        sys.exit(2)
    elif len(args) == 1:
        project_path = args[0]

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-s", "--silent"):
            silent = True

    try:
        fix_icon_imports(project_path, silent)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    if not silent:
        print("Done!")

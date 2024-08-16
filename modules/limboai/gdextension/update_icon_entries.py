#!/usr/bin/python
"""
Usage: update_icon_entries.py [--silent]
Update icon entries in limboai.gdextension file.

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


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))


def update_icon_entries(silent=False):
    config_dir = get_script_dir()
    config_path = os.path.join(config_dir, "limboai.gdextension")
    content = ""
    new_content = ""

    f = open(config_path, "r")
    for line in f:
        content += line
    f.close()

    index = content.find("[icons]")
    new_content = content[0:index]
    new_content += "[icons]\n\n"

    icon_files = []
    icons_dir = os.path.join(config_dir, "../icons/")
    for icon_file in glob.glob(icons_dir + "/*.svg"):
        icon_file = os.path.basename(icon_file)
        icon_files.append(icon_file)

    icon_files.sort()
    for icon_file in icon_files:
        new_content += os.path.splitext(icon_file)[0] + ' = "res://addons/limboai/icons/' + icon_file + '"\n'

    if new_content != content:
        f = open(config_path, "w")
        f.write(new_content)
        f.close()
        if not silent:
            print(new_content)
            print("=== Icon entries updated ===")
    else:
        if not silent:
            print("=== No update needed for icon entries ===")


if __name__ == "__main__":
    silent = False
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

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-s", "--silent"):
            silent = True

    update_icon_entries(silent)

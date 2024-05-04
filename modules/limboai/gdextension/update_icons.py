#!/usr/bin/python
"""
Usage: update_icons.py [--silent]
Update icon declarations in limboai.gdextension file.

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


def main():
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

    config_dir = get_script_dir()
    config_path = os.path.join(config_dir, "limboai.gdextension")
    content = ""

    f = open(config_path, "r")
    for line in f:
        if line.startswith("[icons]"):
            break
        content += line
    f.close()

    content += "[icons]\n\n"

    icon_files = []
    icons_dir = os.path.join(config_dir, "../icons/")
    for icon_file in glob.glob(icons_dir + "/*.svg"):
        icon_file = os.path.basename(icon_file)
        icon_files.append(icon_file)

    icon_files.sort()
    for icon_file in icon_files:
        content += os.path.splitext(icon_file)[0] + ' = "res://addons/limboai/icons/' + icon_file + '"\n'

    f = open(config_path, "w")
    f.write(content)
    f.close()

    if not silent:
        print(content)
        print("======= Icon declarations updated =======")


if __name__ == "__main__":
    main()

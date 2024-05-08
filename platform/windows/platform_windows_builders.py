"""Functions used to generate source files during build time"""

import os
from detect import get_mingw_tool


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if env["separate_debug_symbols"] or os.stat(dst).st_size >= 2040109465:
        objcopy = get_mingw_tool("objcopy", env["mingw_prefix"], env["arch"])
        strip = get_mingw_tool("strip", env["mingw_prefix"], env["arch"])

        if not objcopy or not strip:
            print('`separate_debug_symbols` requires both "objcopy" and "strip" to function.')
            return

        os.system("{0} --only-keep-debug {1} {1}.debugsymbols".format(objcopy, dst))
        os.system("{0} --strip-debug --strip-unneeded {1}".format(strip, dst))
        os.system("{0} --add-gnu-debuglink={1}.debugsymbols {1}".format(objcopy, dst))

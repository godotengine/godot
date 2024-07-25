"""Functions used to generate source files during build time"""

import os

from detect import get_mingw_bin_prefix, try_cmd


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if env["separate_debug_symbols"] or os.stat(dst).st_size >= 2040109465:
        mingw_bin_prefix = get_mingw_bin_prefix(env["mingw_prefix"], env["arch"])
        if try_cmd("objcopy --version", env["mingw_prefix"], env["arch"]):
            os.system(mingw_bin_prefix + "objcopy --only-keep-debug {0} {0}.debugsymbols".format(dst))
        else:
            os.system("objcopy --only-keep-debug {0} {0}.debugsymbols".format(dst))
        if try_cmd("strip --version", env["mingw_prefix"], env["arch"]):
            os.system(mingw_bin_prefix + "strip --strip-debug --strip-unneeded {0}".format(dst))
        else:
            os.system("strip --strip-debug --strip-unneeded {0}".format(dst))
        if try_cmd("objcopy --version", env["mingw_prefix"], env["arch"]):
            os.system(mingw_bin_prefix + "objcopy --add-gnu-debuglink={0}.debugsymbols {0}".format(dst))
        else:
            os.system("objcopy --add-gnu-debuglink={0}.debugsymbols {0}".format(dst))

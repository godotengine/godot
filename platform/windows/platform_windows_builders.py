"""Functions used to generate source files during build time"""

import os


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if env["separate_debug_symbols"] or os.stat(dst).st_size >= 2040109465:
        os.system("{0} --only-keep-debug {1} {1}.debugsymbols".format(env["OBJCOPY"], dst))
        os.system("{0} --strip-debug --strip-unneeded {1}".format(env["STRIP"], dst))
        os.system("{0} --add-gnu-debuglink={1}.debugsymbols {1}".format(env["OBJCOPY"], dst))

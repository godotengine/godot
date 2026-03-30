"""Functions used to generate source files during build time"""

import os


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if env["separate_debug_symbols"] or os.stat(dst).st_size >= 2040109465:
        os.system(f"{env['OBJCOPY']} --only-keep-debug {dst} {dst}.debugsymbols")
        os.system(f"{env['STRIP']} --strip-debug --strip-unneeded {dst}")
        os.system(f"{env['OBJCOPY']} --add-gnu-debuglink={dst}.debugsymbols {dst}")

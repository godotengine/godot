"""Functions used to generate source files during build time"""

import os


def make_debug_linuxbsd(target, source, env):
    dst = str(target[0])
    os.system(f"objcopy --only-keep-debug {dst} {dst}.debugsymbols")
    os.system(f"strip --strip-debug --strip-unneeded {dst}")
    os.system(f"objcopy --add-gnu-debuglink={dst}.debugsymbols {dst}")

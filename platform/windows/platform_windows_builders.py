"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os
from platform_methods import subprocess_main


def make_debug_mingw(target, source, env):
    mingw_prefix = ""
    if env["bits"] == "32":
        mingw_prefix = env["mingw_prefix_32"]
    else:
        mingw_prefix = env["mingw_prefix_64"]
    os.system(mingw_prefix + "objcopy --only-keep-debug {0} {0}.debugsymbols".format(target[0]))
    os.system(mingw_prefix + "strip --strip-debug --strip-unneeded {0}".format(target[0]))
    os.system(mingw_prefix + "objcopy --add-gnu-debuglink={0}.debugsymbols {0}".format(target[0]))


if __name__ == "__main__":
    subprocess_main(globals())

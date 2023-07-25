"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os
from detect import get_mingw_bin_prefix
from detect import try_cmd
from platform_methods import subprocess_main


def make_debug_mingw(target, source, env):
    mingw_bin_prefix = get_mingw_bin_prefix(env["mingw_prefix"], env["arch"])
    if try_cmd("objcopy --version", env["mingw_prefix"], env["arch"]):
        os.system(mingw_bin_prefix + "objcopy --only-keep-debug {0} {0}.debugsymbols".format(target[0]))
    else:
        os.system("objcopy --only-keep-debug {0} {0}.debugsymbols".format(target[0]))
    if try_cmd("strip --version", env["mingw_prefix"], env["arch"]):
        os.system(mingw_bin_prefix + "strip --strip-debug --strip-unneeded {0}".format(target[0]))
    else:
        os.system("strip --strip-debug --strip-unneeded {0}".format(target[0]))
    if try_cmd("objcopy --version", env["mingw_prefix"], env["arch"]):
        os.system(mingw_bin_prefix + "objcopy --add-gnu-debuglink={0}.debugsymbols {0}".format(target[0]))
    else:
        os.system("objcopy --add-gnu-debuglink={0}.debugsymbols {0}".format(target[0]))


if __name__ == "__main__":
    subprocess_main(globals())

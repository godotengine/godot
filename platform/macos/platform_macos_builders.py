"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os
from platform_methods import subprocess_main


def make_debug_macos(target, source, env):
    if env["macports_clang"] != "no":
        mpprefix = os.environ.get("MACPORTS_PREFIX", "/opt/local")
        mpclangver = env["macports_clang"]
        os.system(mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-dsymutil {0} -o {0}.dSYM".format(target[0]))
    else:
        os.system("dsymutil {0} -o {0}.dSYM".format(target[0]))
    os.system("strip -u -r {0}".format(target[0]))


if __name__ == "__main__":
    subprocess_main(globals())

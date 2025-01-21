"""Functions used to generate source files during build time"""

import os


def make_debug_macos(target, source, env):
    dst = str(target[0])
    if env["macports_clang"] != "no":
        mpprefix = os.environ.get("MACPORTS_PREFIX", "/opt/local")
        mpclangver = env["macports_clang"]
        os.system(mpprefix + "/libexec/llvm-" + mpclangver + "/bin/llvm-dsymutil {0} -o {0}.dSYM".format(dst))
    else:
        os.system("dsymutil {0} -o {0}.dSYM".format(dst))
    os.system("strip -u -r {0}".format(dst))

"""Functions used to generate source files during build time"""

import os


def make_debug_macos(target, source, env):
    dst = str(target[0])
    if env["macports_clang"] != "no":
        mpprefix = os.environ.get("MACPORTS_PREFIX", "/opt/local")
        mpclangver = env["macports_clang"]
        os.system(mpprefix + "/libexec/llvm-" + mpclangver + f"/bin/llvm-dsymutil {dst} -o {dst}.dSYM")
    else:
        os.system(f"dsymutil {dst} -o {dst}.dSYM")
    os.system(f"strip -u -r {dst}")

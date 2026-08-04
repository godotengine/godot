"""Functions used to generate source files during build time"""

import os
import subprocess


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if env["separate_debug_symbols"] or os.stat(dst).st_size >= 2040109465:
        subprocess.run([env["OBJCOPY"], "--only-keep-debug", dst, f"{dst}.debugsymbols"], check=True)
        subprocess.run([env["STRIP"], "--strip-debug", "--strip-unneeded", dst], check=True)
        subprocess.run([env["OBJCOPY"], f"--add-gnu-debuglink={dst}.debugsymbols", dst], check=True)

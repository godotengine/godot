"""Functions used to generate source files during build time"""

import os
import subprocess


def make_debug_mingw(target, source, env):
    dst = str(target[0])
    objcopy, strip, separate_debug_symbols = source
    # Force separate debug symbols if executable size is larger than 1.9 GB.
    if separate_debug_symbols or os.stat(dst).st_size >= 2040109465:
        subprocess.run([objcopy, "--only-keep-debug", dst, f"{dst}.debugsymbols"], check=True)
        subprocess.run([strip, "--strip-debug", "--strip-unneeded", dst], check=True)
        subprocess.run([objcopy, f"--add-gnu-debuglink={dst}.debugsymbols", dst], check=True)

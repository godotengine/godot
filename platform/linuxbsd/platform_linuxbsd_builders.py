"""Functions used to generate source files during build time"""

import subprocess


def make_debug_linuxbsd(target, source, env):
    dst = str(target[0])
    subprocess.run(["objcopy", "--only-keep-debug", dst, f"{dst}.debugsymbols"], check=True)
    subprocess.run(["strip", "--strip-debug", "--strip-unneeded", dst], check=True)
    subprocess.run(["objcopy", f"--add-gnu-debuglink={dst}.debugsymbols", dst], check=True)

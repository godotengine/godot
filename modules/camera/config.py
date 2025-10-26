def can_build(env, platform):
    import sys

    if sys.platform.startswith("freebsd") or sys.platform.startswith("openbsd"):
        return False
    return platform == "macos" or platform == "windows" or platform == "linuxbsd" or platform == "android"


def configure(env):
    pass

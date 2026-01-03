def can_build(env, platform):
    import sys

    if (
        sys.platform.startswith("dragonfly")
        or sys.platform.startswith("freebsd")
        or sys.platform.startswith("openbsd")
        or sys.platform.startswith("netbsd")
    ):
        return False
    return platform == "macos" or platform == "windows" or platform == "linuxbsd" or platform == "android"


def configure(env):
    pass

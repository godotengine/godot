def can_build(env, platform):
    import sys

    if sys.platform.startswith("freebsd") or sys.platform.startswith("openbsd"):
        return False
    return platform in ["macos", "windows", "linuxbsd", "android", "web"]


def configure(env):
    pass

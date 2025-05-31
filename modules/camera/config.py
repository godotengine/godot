def can_build(env, platform):
    import sys

    if sys.platform.startswith("freebsd"):
        return False
    return platform in ["macos", "windows", "linuxbsd", "android", "web"]


def configure(env):
    pass

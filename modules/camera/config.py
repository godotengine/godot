def can_build(env, platform):
    import sys

    if sys.platform.startswith("freebsd"):
        return False
    return platform == "macos" or platform == "windows" or platform == "android" or platform == "linuxbsd"


def configure(env):
    pass

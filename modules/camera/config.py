def can_build(env, platform):
    import sys

    if sys.platform.startswith("freebsd") or sys.platform.startswith("openbsd"):
        return False
    return (
        platform == "macos"
        or platform == "windows"
        or platform == "linuxbsd"
        or platform == "android"
        or platform == "ios"
        or platform == "visionos"
    )


def configure(env):
    pass

def can_build(env, platform):
    import sys

    return (
        platform == "macos"
        or platform == "windows"
        or (platform == "linuxbsd" and sys.platform.startswith("linux"))
        or platform == "android"
        or platform == "ios"
        or platform == "visionos"
    )


def configure(env):
    pass

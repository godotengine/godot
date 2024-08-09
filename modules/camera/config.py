def can_build(env, platform):
    return platform == "macos" or platform == "windows" or platform == "linuxbsd"


def configure(env):
    pass

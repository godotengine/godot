def can_build(env, platform):
    return platform == "osx" or platform == "windows" or platform == "linuxbsd" or platform == "x11"


def configure(env):
    pass

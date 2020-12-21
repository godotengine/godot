def can_build(env, platform):
    return env["tools"] and not env["disable_3d"]


def configure(env):
    pass

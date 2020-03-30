def can_build(env, platform):
    return env["tools"] and platform not in ["android", "ios"]


def configure(env):
    pass

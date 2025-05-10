def can_build(env, platform):
    return not env["disable_3d"] and env.editor_build and platform not in ["android", "ios"]


def configure(env):
    pass

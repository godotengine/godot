def can_build(env, platform):
    return not env["disable_3d"] and env.editor_build


def configure(env):
    pass

def can_build(env, platform):
    return not env["disable_3d"] and env.editor_build and env["rendering_device"]


def configure(env):
    pass

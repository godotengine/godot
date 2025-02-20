def can_build(env, platform):
    return env.editor_build and env["rendering_device"] and platform not in ["android", "ios"]


def configure(env):
    pass

def can_build(env, platform):
    return env.module_check_dependencies("msdfgen", ["freetype"])


def configure(env):
    pass

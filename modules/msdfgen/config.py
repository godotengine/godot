def can_build(env, platform):
    env.module_add_dependencies("msdfgen", ["freetype"])
    return True


def configure(env):
    pass

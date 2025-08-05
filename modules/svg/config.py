def can_build(env, platform):
    env.module_add_dependencies("svg", ["jpg", "webp"], True)
    return True


def configure(env):
    pass

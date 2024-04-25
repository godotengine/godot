def can_build(env, platform):
    env.module_add_dependencies("ktx", ["basis_universal"])
    return True


def configure(env):
    pass

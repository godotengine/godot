def can_build(env, platform):
    return env.module_check_dependencies("opus", ["ogg"])


def configure(env):
    pass

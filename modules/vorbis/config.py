def can_build(env, platform):
    return env.module_check_dependencies("vorbis", ["ogg"])


def configure(env):
    pass

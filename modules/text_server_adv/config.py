def can_build(env, platform):
    return env.module_check_dependencies("text_server_adv", ["freetype"])


def configure(env):
    pass

def can_build(env, platform):
    return env.module_check_dependencies("text_server_fb", ["freetype"])


def configure(env):
    pass


def is_enabled():
    # The module is disabled by default. Use module_text_server_fb_enabled=yes to enable it.
    return False

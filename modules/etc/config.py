
def can_build(platform):
    return True


def configure(env):
    # Tools only, disabled for non-tools
    # TODO: Find a cleaner way to achieve that
    if not env['tools']:
        env['module_etc_enabled'] = False
        env.disabled_modules.append("etc")

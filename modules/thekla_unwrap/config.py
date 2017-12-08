def can_build(platform):
    return platform != "android" and platform != "ios"

def configure(env):
    if not env['tools']:
        env['builtin_thekla_atlas'] = False
        env.disabled_modules.append("thekla_unwrap")

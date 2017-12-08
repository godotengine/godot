def can_build(platform):
    return platform != "android" and platform != "ios"

def configure(env):
    if not env['tools']:
        env['module_thekla_unwrap_enabled'] = False
        env.disabled_modules.append("thekla_unwrap")

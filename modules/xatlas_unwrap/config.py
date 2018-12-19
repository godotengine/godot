def can_build(env, platform):
    #return False #xatlas is buggy
    return (env['tools'] and platform not in ["android", "ios"])

def configure(env):
    pass

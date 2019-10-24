def can_build(env, platform):
    return True

def configure(env):
    env.use_ptrcall = True

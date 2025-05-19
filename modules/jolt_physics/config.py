def can_build(env, platform):
    return not env["disable_physics_3d"] and not env["arch"] == "ppc32"


def configure(env):
    pass

def can_build(env, platform):
    # Depends on Embree library, which only supports x86_64 and arm64.
    return env["arch"] in ["x86_64", "arm64"]


def configure(env):
    pass

def can_build(env, platform):
    # Supported architectures and platforms depend on the Embree library.
    return env["arch"] in ["x86_64", "arm64", "wasm32"]


def configure(env):
    pass

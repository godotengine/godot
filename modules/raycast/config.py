def can_build(env, platform):
    # Supported architectures depend on the Embree library.
    if env["arch"] in ["x86_64", "arm64", "wasm32"]:
        return True
    # x86_32 only seems supported on Windows for now.
    if env["arch"] == "x86_32" and platform == "windows":
        return True
    return False


def configure(env):
    pass

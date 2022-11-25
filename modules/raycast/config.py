def can_build(env, platform):
    # Supported architectures depend on the Embree library.
    # No ARM32 support planned.
    if env["arch"] == "arm32":
        return False
    # x86_32 only seems supported on Windows for now.
    if env["arch"] == "x86_32" and platform != "windows":
        return False
    # The rest works, even wasm32!
    return True


def configure(env):
    pass

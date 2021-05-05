def can_build(env, platform):
    if not env["tools"]:
        return False

    # Depends on Embree library, which supports only x86_64 (originally)
    # and aarch64 (thanks to the embree-aarch64 fork).

    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86_64"]

    if platform in ["javascript", "server"]:
        return False

    if env["bits"] == "32":
        return False

    return True


def configure(env):
    pass

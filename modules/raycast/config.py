def can_build(env, platform):
    if not env["tools"]:
        return False

    # Depends on Embree library, which only supports x86_64 and aarch64.
    if env["arch"] in ["arm", "arm32"] or env["arch"].startswith("rv") or env["arch"].startswith("ppc"):
        return False

    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86_64"]

    if platform in ["javascript", "server"]:
        return False

    if env["bits"] == "32":
        return False

    return True


def configure(env):
    pass

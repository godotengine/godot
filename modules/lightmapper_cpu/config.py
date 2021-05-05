def can_build(env, platform):
    if not env["tools"] or not env["module_raycast_enabled"]:
        return False

    # Depends on raycast module (embree), but we can't have access to the result of
    # `can_build()` for that module, so we need to duplicate that code as a short-term
    # solution.

    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86_64"]

    if platform in ["javascript", "server"]:
        return False

    if env["bits"] == "32":
        return False

    return True


def configure(env):
    pass

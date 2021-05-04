def can_build(env, platform):
    if not env["tools"]:
        return False

    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86", "x86_64"]

    if platform in ["javascript", "server"]:
        return False

    return True


def configure(env):
    pass

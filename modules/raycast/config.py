def can_build(env, platform):
    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86", "x86_64"]

    if platform == "javascript":
        return False  # No SIMD support yet

    return True


def configure(env):
    pass

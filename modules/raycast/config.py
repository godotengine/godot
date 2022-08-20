def can_build(env, platform):
    # Depends on Embree library, which only supports x86_64 and aarch64.
    if env["arch"].startswith("rv") or env["arch"].startswith("ppc"):
        return False

    if platform == "android":
        return env["android_arch"] in ["arm64v8", "x86_64"]

    if platform == "javascript":
        return False  # No SIMD support yet

    if env["bits"] == "32":
        return False

    return True


def configure(env):
    pass

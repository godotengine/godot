def can_build(env, platform):
    # Embree requires at least SSE2 to be available, so 32-bit and ARM64 builds are
    # not supported.
    # It's also only relevant for tools build and desktop platforms,
    # as doing lightmap generation on Android or HTML5 would be a bit far-fetched.
    supported_platform = platform in ["x11", "osx", "windows", "server"]
    supported_bits = env["bits"] == "64"
    supported_arch = env["arch"] != "arm64"
    return env["tools"] and supported_platform and supported_bits and supported_arch


def configure(env):
    pass

def can_build(env, platform):
    # Embree requires at least SSE2 to be available, so 32-bit and ARM64 builds are
    # not supported.
    # It's also only relevant for tools build and desktop platforms,
    # as doing lightmap generation on Android or HTML5 would be a bit far-fetched.
    supported_platform = platform in ["x11", "osx", "windows", "server"]
    supported_bits = env["bits"] == "64"
    supported_arch = env["arch"] != "arm64"

    # Hack to disable on Linux arm64. This won't work well for cross-compilation (checks
    # host, not target) and would need a more thorough fix by refactoring our arch and
    # bits-handling code.
    from platform import machine

    if platform == "x11" and machine() != "x86_64":
        supported_arch = False

    return env["tools"] and supported_platform and supported_bits and supported_arch


def configure(env):
    pass

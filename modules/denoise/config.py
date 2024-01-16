def can_build(env, platform):
    # Thirdparty dependency OpenImage Denoise includes oneDNN library
    # and the version we use only supports x86_64.
    # It's also only relevant for tools build and desktop platforms,
    # as doing lightmap generation and denoising on Android or HTML5
    # would be a bit far-fetched.
    # Note: oneDNN doesn't support ARM64, OIDN needs updating to the latest version
    supported_platform = platform in ["x11", "osx", "windows", "server"]
    supported_arch = env["bits"] == "64"
    if env["arch"].startswith("arm"):
        supported_arch = False
    if env["arch"].startswith("ppc"):
        supported_arch = False
    if env["arch"].startswith("rv"):
        supported_arch = False

    # Hack to disable on Linux arm64. This won't work well for cross-compilation (checks
    # host, not target) and would need a more thorough fix by refactoring our arch and
    # bits-handling code.
    from platform import machine

    if platform == "x11" and machine() != "x86_64":
        supported_arch = False

    return env["tools"] and supported_platform and supported_arch


def configure(env):
    pass

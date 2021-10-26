def can_build(env, platform):
    # Thirdparty dependency OpenImage Denoise includes oneDNN library
    # and the version we use only supports x86_64.
    # It's also only relevant for tools build and desktop platforms,
    # as doing lightmap generation and denoising on Android or HTML5
    # would be a bit far-fetched.
    desktop_platforms = ["linuxbsd", "osx", "windows"]
    supported_arch = env["bits"] == "64" and env["arch"] != "arm64" and not env["arch"].startswith("rv")
    return env["tools"] and platform in desktop_platforms and supported_arch


def configure(env):
    pass

def can_build(env: dict) -> bool:
    # Thirdparty dependency OpenImage Denoise includes oneDNN library
    # which only supports 64-bit architectures.
    # It's also only relevant for tools build and desktop platforms,
    # as doing lightmap generation and denoising on Android or HTML5
    # would be a bit far-fetched.
    desktop_platforms = ["linuxbsd", "osx", "windows"]
    return env.tools_enabled and env.platform in desktop_platforms and env.cpu_family == 'x86_64'

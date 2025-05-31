def can_build(env, platform):
    # glslang is only needed when Vulkan, Direct3D 12 or Metal-based renderers are available,
    # as OpenGL doesn't use glslang.
    return env["vulkan"] or env["d3d12"] or env["metal"]


def configure(env):
    pass

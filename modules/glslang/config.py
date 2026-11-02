def can_build(env, platform):
    # glslang is needed for any RenderingDevice backend (Vulkan, D3D12, Metal, WebGPU).
    # OpenGL/GLES3 doesn't use glslang.
    return env["vulkan"] or env["d3d12"] or env["metal"] or env["webgpu"]


def configure(env):
    pass

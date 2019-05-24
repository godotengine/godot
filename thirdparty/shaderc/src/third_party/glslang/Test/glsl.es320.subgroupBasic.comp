#version 320 es

#extension GL_KHR_shader_subgroup_basic: enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) buffer Buffer
{
    int a[];
} data;

void main()
{
    data.a[gl_SubgroupSize] = 1;
    data.a[gl_SubgroupInvocationID] = 1;
    data.a[gl_NumSubgroups] = 1;
    data.a[gl_SubgroupID] = (subgroupElect()) ? 1 : 0;
    subgroupBarrier();
    subgroupMemoryBarrier();
    subgroupMemoryBarrierBuffer();
    subgroupMemoryBarrierShared();
    subgroupMemoryBarrierImage();
}

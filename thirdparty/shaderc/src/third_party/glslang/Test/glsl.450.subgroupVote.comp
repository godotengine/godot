#version 450

#extension GL_KHR_shader_subgroup_vote: enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) buffer Buffers
{
    vec4  f4;
    ivec4 i4;
    uvec4 u4;
    dvec4 d4;
    int r;
} data[4];

void main()
{
    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4;

    if (subgroupAll(data[invocation].r < 0))
    {
        data[invocation].r = int(subgroupAllEqual(data[0].f4.x));
        data[invocation].r = int(subgroupAllEqual(data[1].f4.xy));
        data[invocation].r = int(subgroupAllEqual(data[2].f4.xyz));
        data[invocation].r = int(subgroupAllEqual(data[3].f4));

        data[invocation].r = int(subgroupAllEqual(data[0].i4.x));
        data[invocation].r = int(subgroupAllEqual(data[1].i4.xy));
        data[invocation].r = int(subgroupAllEqual(data[2].i4.xyz));
        data[invocation].r = int(subgroupAllEqual(data[3].i4));

        data[invocation].r = int(subgroupAllEqual(data[0].u4.x));
        data[invocation].r = int(subgroupAllEqual(data[1].u4.xy));
        data[invocation].r = int(subgroupAllEqual(data[2].u4.xyz));
        data[invocation].r = int(subgroupAllEqual(data[3].u4));
    }
    else if (subgroupAny(data[invocation].r < 0))
    {
        data[invocation].r = int(subgroupAllEqual(data[0].d4.x));
        data[invocation].r = int(subgroupAllEqual(data[1].d4.xy));
        data[invocation].r = int(subgroupAllEqual(data[2].d4.xyz));
        data[invocation].r = int(subgroupAllEqual(data[3].d4));

        data[invocation].r = int(int(subgroupAllEqual(data[0].i4.x < 0)));
        data[invocation].r = int(ivec2(subgroupAllEqual(lessThan(data[1].i4.xy, ivec2(0)))));
        data[invocation].r = int(ivec3(subgroupAllEqual(lessThan(data[1].i4.xyz, ivec3(0)))));
        data[invocation].r = int(ivec4(subgroupAllEqual(lessThan(data[1].i4, ivec4(0)))));
    }
}

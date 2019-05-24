#version 320 es

#extension GL_KHR_shader_subgroup_clustered: enable

layout (local_size_x = 8) in;

layout(binding = 0) buffer Buffers
{
    vec4  f4;
    ivec4 i4;
    uvec4 u4;
} data[4];

void main()
{
    int a = 1;
    const int aConst = 1;

    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4u;

    data[0].f4.xy  = subgroupClusteredAdd(data[1].f4.xy, 0u);          // ERROR, less than 1

    data[0].f4.x   = subgroupClusteredMul(data[0].f4.x, 3u);           // ERROR, not a power of 2

    data[1].i4.xy  = subgroupClusteredMin(data[1].i4.xy, 8u);
    data[1].i4.xyz = subgroupClusteredMin(data[2].i4.xyz, 6u);         // ERROR, not a power of 2

    data[3].i4.x   = subgroupClusteredOr(data[0].i4.x, uint(a));            // ERROR, not constant
    data[3].i4.xy  = subgroupClusteredOr(data[1].i4.xy, uint(aConst));

    data[0].i4.x   = subgroupClusteredXor(data[0].i4.x, uint(1 + a));       // ERROR, not constant
    data[0].i4.xy  = subgroupClusteredXor(data[1].i4.xy, uint(aConst + a)); // ERROR, not constant
    data[0].i4.xyz = subgroupClusteredXor(data[2].i4.xyz, uint(1 + aConst));
}

#version 450

#extension GL_KHR_shader_subgroup_clustered: enable

layout (local_size_x = 8) in;

layout(binding = 0) buffer Buffers
{
    vec4  f4;
    ivec4 i4;
    uvec4 u4;
    dvec4 d4;
} data[4];

void main()
{
    int a = 1;
    const int aConst = 1;

    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4;

    data[invocation].f4.xy  = subgroupClusteredAdd(data[1].f4.xy, 0);          // ERROR, less than 1

    data[invocation].f4.x   = subgroupClusteredMul(data[0].f4.x, 3);           // ERROR, not a power of 2

    data[invocation].i4.xy  = subgroupClusteredMin(data[1].i4.xy, 8);
    data[invocation].i4.xyz = subgroupClusteredMin(data[2].i4.xyz, 6);         // ERROR, not a power of 2

    data[invocation].f4.x   = subgroupClusteredMax(data[0].f4.x, -1);          // ERROR, less than 1

    data[invocation].i4     = subgroupClusteredAnd(data[3].i4, -3);            // ERROR, less than 1

    data[invocation].i4.x   = subgroupClusteredOr(data[0].i4.x, a);            // ERROR, not constant
    data[invocation].i4.xy  = subgroupClusteredOr(data[1].i4.xy, aConst);

    data[invocation].i4.x   = subgroupClusteredXor(data[0].i4.x, 1 + a);       // ERROR, not constant
    data[invocation].i4.xy  = subgroupClusteredXor(data[1].i4.xy, aConst + a); // ERROR, not constant
    data[invocation].i4.xyz = subgroupClusteredXor(data[2].i4.xyz, 1 + aConst);
}

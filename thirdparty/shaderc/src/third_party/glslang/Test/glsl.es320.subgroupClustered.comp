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
    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4u;

    data[0].f4.x   = subgroupClusteredAdd(data[0].f4.x, 1u);
    data[0].f4.xy  = subgroupClusteredAdd(data[1].f4.xy, 1u);
    data[0].f4.xyz = subgroupClusteredAdd(data[2].f4.xyz, 1u);
    data[0].f4     = subgroupClusteredAdd(data[3].f4, 1u);

    data[1].i4.x   = subgroupClusteredAdd(data[0].i4.x, 1u);
    data[1].i4.xy  = subgroupClusteredAdd(data[1].i4.xy, 1u);
    data[1].i4.xyz = subgroupClusteredAdd(data[2].i4.xyz, 1u);
    data[1].i4     = subgroupClusteredAdd(data[3].i4, 1u);

    data[2].u4.x   = subgroupClusteredAdd(data[0].u4.x, 1u);
    data[2].u4.xy  = subgroupClusteredAdd(data[1].u4.xy, 1u);
    data[2].u4.xyz = subgroupClusteredAdd(data[2].u4.xyz, 1u);
    data[2].u4     = subgroupClusteredAdd(data[3].u4, 1u);

    data[3].f4.x   = subgroupClusteredMul(data[0].f4.x, 1u);
    data[3].f4.xy  = subgroupClusteredMul(data[1].f4.xy, 1u);
    data[3].f4.xyz = subgroupClusteredMul(data[2].f4.xyz, 1u);
    data[3].f4     = subgroupClusteredMul(data[3].f4, 1u);

    data[0].i4.x   = subgroupClusteredMul(data[0].i4.x, 1u);
    data[0].i4.xy  = subgroupClusteredMul(data[1].i4.xy, 1u);
    data[0].i4.xyz = subgroupClusteredMul(data[2].i4.xyz, 1u);
    data[0].i4     = subgroupClusteredMul(data[3].i4, 1u);

    data[1].u4.x   = subgroupClusteredMul(data[0].u4.x, 1u);
    data[1].u4.xy  = subgroupClusteredMul(data[1].u4.xy, 1u);
    data[1].u4.xyz = subgroupClusteredMul(data[2].u4.xyz, 1u);
    data[1].u4     = subgroupClusteredMul(data[3].u4, 1u);

    data[2].f4.x   = subgroupClusteredMin(data[0].f4.x, 1u);
    data[2].f4.xy  = subgroupClusteredMin(data[1].f4.xy, 1u);
    data[2].f4.xyz = subgroupClusteredMin(data[2].f4.xyz, 1u);
    data[2].f4     = subgroupClusteredMin(data[3].f4, 1u);

    data[3].i4.x   = subgroupClusteredMin(data[0].i4.x, 1u);
    data[3].i4.xy  = subgroupClusteredMin(data[1].i4.xy, 1u);
    data[3].i4.xyz = subgroupClusteredMin(data[2].i4.xyz, 1u);
    data[3].i4     = subgroupClusteredMin(data[3].i4, 1u);

    data[0].u4.x   = subgroupClusteredMin(data[0].u4.x, 1u);
    data[0].u4.xy  = subgroupClusteredMin(data[1].u4.xy, 1u);
    data[0].u4.xyz = subgroupClusteredMin(data[2].u4.xyz, 1u);
    data[0].u4     = subgroupClusteredMin(data[3].u4, 1u);

    data[1].f4.x   = subgroupClusteredMax(data[0].f4.x, 1u);
    data[1].f4.xy  = subgroupClusteredMax(data[1].f4.xy, 1u);
    data[1].f4.xyz = subgroupClusteredMax(data[2].f4.xyz, 1u);
    data[1].f4     = subgroupClusteredMax(data[3].f4, 1u);

    data[2].i4.x   = subgroupClusteredMax(data[0].i4.x, 1u);
    data[2].i4.xy  = subgroupClusteredMax(data[1].i4.xy, 1u);
    data[2].i4.xyz = subgroupClusteredMax(data[2].i4.xyz, 1u);
    data[2].i4     = subgroupClusteredMax(data[3].i4, 1u);

    data[3].u4.x   = subgroupClusteredMax(data[0].u4.x, 1u);
    data[3].u4.xy  = subgroupClusteredMax(data[1].u4.xy, 1u);
    data[3].u4.xyz = subgroupClusteredMax(data[2].u4.xyz, 1u);
    data[3].u4     = subgroupClusteredMax(data[3].u4, 1u);

    data[0].i4.x   = subgroupClusteredAnd(data[0].i4.x, 1u);
    data[0].i4.xy  = subgroupClusteredAnd(data[1].i4.xy, 1u);
    data[0].i4.xyz = subgroupClusteredAnd(data[2].i4.xyz, 1u);
    data[0].i4     = subgroupClusteredAnd(data[3].i4, 1u);

    data[1].u4.x   = subgroupClusteredAnd(data[0].u4.x, 1u);
    data[1].u4.xy  = subgroupClusteredAnd(data[1].u4.xy, 1u);
    data[1].u4.xyz = subgroupClusteredAnd(data[2].u4.xyz, 1u);
    data[1].u4     = subgroupClusteredAnd(data[3].u4, 1u);

    data[2].i4.x   =   int(subgroupClusteredAnd(data[0].i4.x < 0, 1u));
    data[2].i4.xy  = ivec2(subgroupClusteredAnd(lessThan(data[1].i4.xy, ivec2(0)), 1u));
    data[2].i4.xyz = ivec3(subgroupClusteredAnd(lessThan(data[1].i4.xyz, ivec3(0)), 1u));
    data[2].i4     = ivec4(subgroupClusteredAnd(lessThan(data[1].i4, ivec4(0)), 1u));

    data[3].i4.x   = subgroupClusteredOr(data[0].i4.x, 1u);
    data[3].i4.xy  = subgroupClusteredOr(data[1].i4.xy, 1u);
    data[3].i4.xyz = subgroupClusteredOr(data[2].i4.xyz, 1u);
    data[3].i4     = subgroupClusteredOr(data[3].i4, 1u);

    data[0].u4.x   = subgroupClusteredOr(data[0].u4.x, 1u);
    data[0].u4.xy  = subgroupClusteredOr(data[1].u4.xy, 1u);
    data[0].u4.xyz = subgroupClusteredOr(data[2].u4.xyz, 1u);
    data[0].u4     = subgroupClusteredOr(data[3].u4, 1u);

    data[1].i4.x   =   int(subgroupClusteredOr(data[0].i4.x < 0, 1u));
    data[1].i4.xy  = ivec2(subgroupClusteredOr(lessThan(data[1].i4.xy, ivec2(0)), 1u));
    data[1].i4.xyz = ivec3(subgroupClusteredOr(lessThan(data[1].i4.xyz, ivec3(0)), 1u));
    data[1].i4     = ivec4(subgroupClusteredOr(lessThan(data[1].i4, ivec4(0)), 1u));

    data[2].i4.x   = subgroupClusteredXor(data[0].i4.x, 1u);
    data[2].i4.xy  = subgroupClusteredXor(data[1].i4.xy, 1u);
    data[2].i4.xyz = subgroupClusteredXor(data[2].i4.xyz, 1u);
    data[2].i4     = subgroupClusteredXor(data[3].i4, 1u);

    data[3].u4.x   = subgroupClusteredXor(data[0].u4.x, 1u);
    data[3].u4.xy  = subgroupClusteredXor(data[1].u4.xy, 1u);
    data[3].u4.xyz = subgroupClusteredXor(data[2].u4.xyz, 1u);
    data[3].u4     = subgroupClusteredXor(data[3].u4, 1u);

    data[0].i4.x   =   int(subgroupClusteredXor(data[0].i4.x < 0, 1u));
    data[0].i4.xy  = ivec2(subgroupClusteredXor(lessThan(data[1].i4.xy, ivec2(0)), 1u));
    data[0].i4.xyz = ivec3(subgroupClusteredXor(lessThan(data[1].i4.xyz, ivec3(0)), 1u));
    data[0].i4     = ivec4(subgroupClusteredXor(lessThan(data[1].i4, ivec4(0)), 1u));
}

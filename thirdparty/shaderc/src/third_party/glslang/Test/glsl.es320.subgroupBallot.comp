#version 320 es

#extension GL_KHR_shader_subgroup_ballot: enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) buffer Buffers
{
    vec4  f4;
    ivec4 i4;
    uvec4 u4;
} data[4];

void main()
{
    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4u;

    uvec4 relMask = gl_SubgroupEqMask +
                       gl_SubgroupGeMask +
                       gl_SubgroupGtMask +
                       gl_SubgroupLeMask +
                       gl_SubgroupLtMask;

    uvec4 result = subgroupBallot(true);

    data[0].u4.x = subgroupBallotBitCount(result);
    data[0].u4.y = subgroupBallotBitExtract(result, 0u) ? 1u : 0u;
    data[0].u4.z = subgroupBallotInclusiveBitCount(result) + subgroupBallotExclusiveBitCount(result);
    data[0].u4.w = subgroupBallotFindLSB(result) + subgroupBallotFindMSB(result);

    if ((relMask == result) && subgroupInverseBallot(data[0].u4))
    {
        data[1].f4.x   = subgroupBroadcast(data[0].f4.x,    3u);
        data[1].f4.xy  = subgroupBroadcast(data[1].f4.xy,   3u);
        data[1].f4.xyz = subgroupBroadcast(data[2].f4.xyz,  3u);
        data[1].f4     = subgroupBroadcast(data[3].f4,      3u);

        data[2].i4.x   = subgroupBroadcast(data[0].i4.x,    2u);
        data[2].i4.xy  = subgroupBroadcast(data[1].i4.xy,   2u);
        data[2].i4.xyz = subgroupBroadcast(data[2].i4.xyz,  2u);
        data[2].i4     = subgroupBroadcast(data[3].i4,      2u);

        data[3].u4.x   = subgroupBroadcast(data[0].u4.x,    1u);
        data[3].u4.xy  = subgroupBroadcast(data[1].u4.xy,   1u);
        data[3].u4.xyz = subgroupBroadcast(data[2].u4.xyz,  1u);
        data[3].u4     = subgroupBroadcast(data[3].u4,      1u);

        data[0].i4.x   = int(subgroupBroadcast(data[0].i4.x < 0,            1u));
        data[0].i4.xy  = ivec2(subgroupBroadcast(lessThan(data[1].i4.xy, ivec2(0)), 1u));
        data[0].i4.xyz = ivec3(subgroupBroadcast(lessThan(data[1].i4.xyz, ivec3(0)), 1u));
        data[0].i4     = ivec4(subgroupBroadcast(lessThan(data[1].i4, ivec4(0)), 1u));
    }
    else
    {
        data[1].f4.x   = subgroupBroadcastFirst(data[0].f4.x);
        data[1].f4.xy  = subgroupBroadcastFirst(data[1].f4.xy);
        data[1].f4.xyz = subgroupBroadcastFirst(data[2].f4.xyz);
        data[1].f4     = subgroupBroadcastFirst(data[3].f4);

        data[2].i4.x   = subgroupBroadcastFirst(data[0].i4.x);
        data[2].i4.xy  = subgroupBroadcastFirst(data[1].i4.xy);
        data[2].i4.xyz = subgroupBroadcastFirst(data[2].i4.xyz);
        data[2].i4     = subgroupBroadcastFirst(data[3].i4);

        data[3].u4.x   = subgroupBroadcastFirst(data[0].u4.x);
        data[3].u4.xy  = subgroupBroadcastFirst(data[1].u4.xy);
        data[3].u4.xyz = subgroupBroadcastFirst(data[2].u4.xyz);
        data[3].u4     = subgroupBroadcastFirst(data[3].u4);

        data[0].i4.x   = int(subgroupBroadcastFirst(data[0].i4.x < 0));
        data[0].i4.xy  = ivec2(subgroupBroadcastFirst(lessThan(data[1].i4.xy, ivec2(0))));
        data[0].i4.xyz = ivec3(subgroupBroadcastFirst(lessThan(data[1].i4.xyz, ivec3(0))));
        data[0].i4     = ivec4(subgroupBroadcastFirst(lessThan(data[1].i4, ivec4(0))));
    }
}

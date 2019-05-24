#version 450

#extension GL_KHR_shader_subgroup_ballot: enable

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) buffer Buffers
{
    vec4  f4;
    ivec4 i4;
    uvec4 u4;
    dvec4 d4;
} data[4];

void main()
{
    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4;

    uvec4 relMask = gl_SubgroupEqMask +
                       gl_SubgroupGeMask +
                       gl_SubgroupGtMask +
                       gl_SubgroupLeMask +
                       gl_SubgroupLtMask;

    uvec4 result = subgroupBallot(true);

    data[invocation].u4.x = subgroupBallotBitCount(result);
    data[invocation].u4.y = subgroupBallotBitExtract(result, 0) ? 1 : 0;
    data[invocation].u4.z = subgroupBallotInclusiveBitCount(result) + subgroupBallotExclusiveBitCount(result);
    data[invocation].u4.w = subgroupBallotFindLSB(result) + subgroupBallotFindMSB(result);

    if ((relMask == result) && subgroupInverseBallot(data[0].u4))
    {
        data[invocation].f4.x   = subgroupBroadcast(data[0].f4.x,    3);
        data[invocation].f4.xy  = subgroupBroadcast(data[1].f4.xy,   3);
        data[invocation].f4.xyz = subgroupBroadcast(data[2].f4.xyz,  3);
        data[invocation].f4     = subgroupBroadcast(data[3].f4,      3);

        data[invocation].i4.x   = subgroupBroadcast(data[0].i4.x,    2);
        data[invocation].i4.xy  = subgroupBroadcast(data[1].i4.xy,   2);
        data[invocation].i4.xyz = subgroupBroadcast(data[2].i4.xyz,  2);
        data[invocation].i4     = subgroupBroadcast(data[3].i4,      2);

        data[invocation].u4.x   = subgroupBroadcast(data[0].u4.x,    1);
        data[invocation].u4.xy  = subgroupBroadcast(data[1].u4.xy,   1);
        data[invocation].u4.xyz = subgroupBroadcast(data[2].u4.xyz,  1);
        data[invocation].u4     = subgroupBroadcast(data[3].u4,      1);

        data[invocation].d4.x   = subgroupBroadcast(data[0].d4.x,    0);
        data[invocation].d4.xy  = subgroupBroadcast(data[1].d4.xy,   0);
        data[invocation].d4.xyz = subgroupBroadcast(data[2].d4.xyz,  0);
        data[invocation].d4     = subgroupBroadcast(data[3].d4,      0);

        data[invocation].i4.x   = int(subgroupBroadcast(data[0].i4.x < 0,            1));
        data[invocation].i4.xy  = ivec2(subgroupBroadcast(lessThan(data[1].i4.xy, ivec2(0)), 1));
        data[invocation].i4.xyz = ivec3(subgroupBroadcast(lessThan(data[1].i4.xyz, ivec3(0)), 1));
        data[invocation].i4     = ivec4(subgroupBroadcast(lessThan(data[1].i4, ivec4(0)), 1));
    }
    else
    {
        data[invocation].f4.x   = subgroupBroadcastFirst(data[0].f4.x);
        data[invocation].f4.xy  = subgroupBroadcastFirst(data[1].f4.xy);
        data[invocation].f4.xyz = subgroupBroadcastFirst(data[2].f4.xyz);
        data[invocation].f4     = subgroupBroadcastFirst(data[3].f4);

        data[invocation].i4.x   = subgroupBroadcastFirst(data[0].i4.x);
        data[invocation].i4.xy  = subgroupBroadcastFirst(data[1].i4.xy);
        data[invocation].i4.xyz = subgroupBroadcastFirst(data[2].i4.xyz);
        data[invocation].i4     = subgroupBroadcastFirst(data[3].i4);

        data[invocation].u4.x   = subgroupBroadcastFirst(data[0].u4.x);
        data[invocation].u4.xy  = subgroupBroadcastFirst(data[1].u4.xy);
        data[invocation].u4.xyz = subgroupBroadcastFirst(data[2].u4.xyz);
        data[invocation].u4     = subgroupBroadcastFirst(data[3].u4);

        data[invocation].d4.x   = subgroupBroadcastFirst(data[0].d4.x);
        data[invocation].d4.xy  = subgroupBroadcastFirst(data[1].d4.xy);
        data[invocation].d4.xyz = subgroupBroadcastFirst(data[2].d4.xyz);
        data[invocation].d4     = subgroupBroadcastFirst(data[3].d4);

        data[invocation].i4.x   = int(subgroupBroadcastFirst(data[0].i4.x < 0));
        data[invocation].i4.xy  = ivec2(subgroupBroadcastFirst(lessThan(data[1].i4.xy, ivec2(0))));
        data[invocation].i4.xyz = ivec3(subgroupBroadcastFirst(lessThan(data[1].i4.xyz, ivec3(0))));
        data[invocation].i4     = ivec4(subgroupBroadcastFirst(lessThan(data[1].i4, ivec4(0))));
    }
}

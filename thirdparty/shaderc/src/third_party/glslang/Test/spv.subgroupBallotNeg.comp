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

    data[invocation].f4.x   = subgroupBroadcast(data[0].f4.x,    invocation);  // ERROR: not constant
}

#version 450

#extension GL_NV_shader_subgroup_partitioned: enable

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
    uint invocation = (gl_SubgroupInvocationID + gl_SubgroupSize) % 4;

    uvec4 ballot = subgroupPartitionNV(invocation);

    data[invocation].u4 = subgroupPartitionNV(data[0].f4.x);
    data[invocation].u4 = subgroupPartitionNV(data[0].f4.xy);
    data[invocation].u4 = subgroupPartitionNV(data[0].f4.xyz);
    data[invocation].u4 = subgroupPartitionNV(data[0].f4);

    data[invocation].u4 = subgroupPartitionNV(data[0].i4.x);
    data[invocation].u4 = subgroupPartitionNV(data[0].i4.xy);
    data[invocation].u4 = subgroupPartitionNV(data[0].i4.xyz);
    data[invocation].u4 = subgroupPartitionNV(data[0].i4);

    data[invocation].u4 = subgroupPartitionNV(data[0].u4.x);
    data[invocation].u4 = subgroupPartitionNV(data[0].u4.xy);
    data[invocation].u4 = subgroupPartitionNV(data[0].u4.xyz);
    data[invocation].u4 = subgroupPartitionNV(data[0].u4);

    data[invocation].u4 = subgroupPartitionNV(data[0].d4.x);
    data[invocation].u4 = subgroupPartitionNV(data[0].d4.xy);
    data[invocation].u4 = subgroupPartitionNV(data[0].d4.xyz);
    data[invocation].u4 = subgroupPartitionNV(data[0].d4);

    data[invocation].u4 = subgroupPartitionNV(bool(data[0].i4.x));
    data[invocation].u4 = subgroupPartitionNV(bvec2(data[0].i4.xy));
    data[invocation].u4 = subgroupPartitionNV(bvec3(data[0].i4.xyz));
    data[invocation].u4 = subgroupPartitionNV(bvec4(data[0].i4));

    data[invocation].f4.x   = subgroupPartitionedAddNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedAddNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedAddNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedAddNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedAddNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedAddNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedAddNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedAddNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedAddNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedAddNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedAddNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedAddNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedAddNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedAddNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedAddNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedAddNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedMulNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedMulNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedMulNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedMulNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedMulNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedMulNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedMulNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedMulNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedMulNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedMulNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedMulNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedMulNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedMulNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedMulNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedMulNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedMulNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedMinNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedMinNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedMinNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedMinNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedMinNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedMinNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedMinNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedMinNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedMinNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedMinNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedMinNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedMinNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedMinNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedMinNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedMinNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedMinNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedMaxNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedMaxNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedMaxNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedMaxNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedMaxNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedMaxNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedMaxNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedMaxNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedMaxNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedMaxNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedMaxNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedMaxNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedMaxNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedMaxNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedMaxNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedMaxNV(data[3].d4, ballot);

    data[invocation].i4.x   = subgroupPartitionedAndNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedAndNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedAndNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedAndNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedAndNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedAndNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedAndNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedAndNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedAndNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedAndNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedAndNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedAndNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedOrNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedOrNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedOrNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedOrNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedOrNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedOrNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedOrNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedOrNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedOrNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedOrNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedOrNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedOrNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedXorNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedXorNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedXorNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedXorNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedXorNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedXorNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedXorNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedXorNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedXorNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedXorNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedXorNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedXorNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].f4.x   = subgroupPartitionedInclusiveAddNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedInclusiveAddNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedInclusiveAddNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedInclusiveAddNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedInclusiveAddNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveAddNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveAddNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveAddNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveAddNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveAddNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveAddNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveAddNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedInclusiveAddNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedInclusiveAddNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedInclusiveAddNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedInclusiveAddNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedInclusiveMulNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedInclusiveMulNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedInclusiveMulNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedInclusiveMulNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedInclusiveMulNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveMulNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveMulNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveMulNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveMulNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveMulNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveMulNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveMulNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedInclusiveMulNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedInclusiveMulNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedInclusiveMulNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedInclusiveMulNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedInclusiveMinNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedInclusiveMinNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedInclusiveMinNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedInclusiveMinNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedInclusiveMinNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveMinNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveMinNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveMinNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveMinNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveMinNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveMinNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveMinNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedInclusiveMinNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedInclusiveMinNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedInclusiveMinNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedInclusiveMinNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedInclusiveMaxNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedInclusiveMaxNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedInclusiveMaxNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedInclusiveMaxNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedInclusiveMaxNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveMaxNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveMaxNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveMaxNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveMaxNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveMaxNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveMaxNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveMaxNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedInclusiveMaxNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedInclusiveMaxNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedInclusiveMaxNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedInclusiveMaxNV(data[3].d4, ballot);

    data[invocation].i4.x   = subgroupPartitionedInclusiveAndNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveAndNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveAndNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveAndNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveAndNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveAndNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveAndNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveAndNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedInclusiveAndNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedInclusiveAndNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedInclusiveAndNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedInclusiveAndNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedInclusiveOrNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveOrNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveOrNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveOrNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveOrNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveOrNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveOrNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveOrNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedInclusiveOrNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedInclusiveOrNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedInclusiveOrNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedInclusiveOrNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedInclusiveXorNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedInclusiveXorNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedInclusiveXorNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedInclusiveXorNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedInclusiveXorNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedInclusiveXorNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedInclusiveXorNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedInclusiveXorNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedInclusiveXorNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedInclusiveXorNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedInclusiveXorNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedInclusiveXorNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].f4.x   = subgroupPartitionedExclusiveAddNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedExclusiveAddNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedExclusiveAddNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedExclusiveAddNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedExclusiveAddNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveAddNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveAddNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveAddNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveAddNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveAddNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveAddNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveAddNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedExclusiveAddNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedExclusiveAddNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedExclusiveAddNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedExclusiveAddNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedExclusiveMulNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedExclusiveMulNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedExclusiveMulNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedExclusiveMulNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedExclusiveMulNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveMulNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveMulNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveMulNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveMulNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveMulNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveMulNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveMulNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedExclusiveMulNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedExclusiveMulNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedExclusiveMulNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedExclusiveMulNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedExclusiveMinNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedExclusiveMinNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedExclusiveMinNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedExclusiveMinNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedExclusiveMinNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveMinNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveMinNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveMinNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveMinNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveMinNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveMinNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveMinNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedExclusiveMinNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedExclusiveMinNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedExclusiveMinNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedExclusiveMinNV(data[3].d4, ballot);

    data[invocation].f4.x   = subgroupPartitionedExclusiveMaxNV(data[0].f4.x, ballot);
    data[invocation].f4.xy  = subgroupPartitionedExclusiveMaxNV(data[1].f4.xy, ballot);
    data[invocation].f4.xyz = subgroupPartitionedExclusiveMaxNV(data[2].f4.xyz, ballot);
    data[invocation].f4     = subgroupPartitionedExclusiveMaxNV(data[3].f4, ballot);

    data[invocation].i4.x   = subgroupPartitionedExclusiveMaxNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveMaxNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveMaxNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveMaxNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveMaxNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveMaxNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveMaxNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveMaxNV(data[3].u4, ballot);

    data[invocation].d4.x   = subgroupPartitionedExclusiveMaxNV(data[0].d4.x, ballot);
    data[invocation].d4.xy  = subgroupPartitionedExclusiveMaxNV(data[1].d4.xy, ballot);
    data[invocation].d4.xyz = subgroupPartitionedExclusiveMaxNV(data[2].d4.xyz, ballot);
    data[invocation].d4     = subgroupPartitionedExclusiveMaxNV(data[3].d4, ballot);

    data[invocation].i4.x   = subgroupPartitionedExclusiveAndNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveAndNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveAndNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveAndNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveAndNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveAndNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveAndNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveAndNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedExclusiveAndNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedExclusiveAndNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedExclusiveAndNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedExclusiveAndNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedExclusiveOrNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveOrNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveOrNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveOrNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveOrNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveOrNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveOrNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveOrNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedExclusiveOrNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedExclusiveOrNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedExclusiveOrNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedExclusiveOrNV(lessThan(data[1].i4, ivec4(0)), ballot));

    data[invocation].i4.x   = subgroupPartitionedExclusiveXorNV(data[0].i4.x, ballot);
    data[invocation].i4.xy  = subgroupPartitionedExclusiveXorNV(data[1].i4.xy, ballot);
    data[invocation].i4.xyz = subgroupPartitionedExclusiveXorNV(data[2].i4.xyz, ballot);
    data[invocation].i4     = subgroupPartitionedExclusiveXorNV(data[3].i4, ballot);

    data[invocation].u4.x   = subgroupPartitionedExclusiveXorNV(data[0].u4.x, ballot);
    data[invocation].u4.xy  = subgroupPartitionedExclusiveXorNV(data[1].u4.xy, ballot);
    data[invocation].u4.xyz = subgroupPartitionedExclusiveXorNV(data[2].u4.xyz, ballot);
    data[invocation].u4     = subgroupPartitionedExclusiveXorNV(data[3].u4, ballot);

    data[invocation].i4.x   =   int(subgroupPartitionedExclusiveXorNV(data[0].i4.x < 0, ballot));
    data[invocation].i4.xy  = ivec2(subgroupPartitionedExclusiveXorNV(lessThan(data[1].i4.xy, ivec2(0)), ballot));
    data[invocation].i4.xyz = ivec3(subgroupPartitionedExclusiveXorNV(lessThan(data[1].i4.xyz, ivec3(0)), ballot));
    data[invocation].i4     = ivec4(subgroupPartitionedExclusiveXorNV(lessThan(data[1].i4, ivec4(0)), ballot));
}

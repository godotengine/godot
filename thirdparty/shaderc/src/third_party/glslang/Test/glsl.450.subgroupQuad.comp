#version 450

#extension GL_KHR_shader_subgroup_quad: enable

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

    data[invocation].f4.x   = subgroupQuadBroadcast(data[0].f4.x, 1);
    data[invocation].f4.xy  = subgroupQuadBroadcast(data[1].f4.xy, 1);
    data[invocation].f4.xyz = subgroupQuadBroadcast(data[2].f4.xyz, 1);
    data[invocation].f4     = subgroupQuadBroadcast(data[3].f4, 1);

    data[invocation].i4.x   = subgroupQuadBroadcast(data[0].i4.x, 1);
    data[invocation].i4.xy  = subgroupQuadBroadcast(data[1].i4.xy, 1);
    data[invocation].i4.xyz = subgroupQuadBroadcast(data[2].i4.xyz, 1);
    data[invocation].i4     = subgroupQuadBroadcast(data[3].i4, 1);

    data[invocation].u4.x   = subgroupQuadBroadcast(data[0].u4.x, 1);
    data[invocation].u4.xy  = subgroupQuadBroadcast(data[1].u4.xy, 1);
    data[invocation].u4.xyz = subgroupQuadBroadcast(data[2].u4.xyz, 1);
    data[invocation].u4     = subgroupQuadBroadcast(data[3].u4, 1);

    data[invocation].d4.x   = subgroupQuadBroadcast(data[0].d4.x, 1);
    data[invocation].d4.xy  = subgroupQuadBroadcast(data[1].d4.xy, 1);
    data[invocation].d4.xyz = subgroupQuadBroadcast(data[2].d4.xyz, 1);
    data[invocation].d4     = subgroupQuadBroadcast(data[3].d4, 1);

    data[invocation].i4.x   =   int(subgroupQuadBroadcast(data[0].i4.x < 0, 1));
    data[invocation].i4.xy  = ivec2(subgroupQuadBroadcast(lessThan(data[1].i4.xy, ivec2(0)), 1));
    data[invocation].i4.xyz = ivec3(subgroupQuadBroadcast(lessThan(data[1].i4.xyz, ivec3(0)), 1));
    data[invocation].i4     = ivec4(subgroupQuadBroadcast(lessThan(data[1].i4, ivec4(0)), 1));

    data[invocation].f4.x   = subgroupQuadSwapHorizontal(data[0].f4.x);
    data[invocation].f4.xy  = subgroupQuadSwapHorizontal(data[1].f4.xy);
    data[invocation].f4.xyz = subgroupQuadSwapHorizontal(data[2].f4.xyz);
    data[invocation].f4     = subgroupQuadSwapHorizontal(data[3].f4);

    data[invocation].i4.x   = subgroupQuadSwapHorizontal(data[0].i4.x);
    data[invocation].i4.xy  = subgroupQuadSwapHorizontal(data[1].i4.xy);
    data[invocation].i4.xyz = subgroupQuadSwapHorizontal(data[2].i4.xyz);
    data[invocation].i4     = subgroupQuadSwapHorizontal(data[3].i4);

    data[invocation].u4.x   = subgroupQuadSwapHorizontal(data[0].u4.x);
    data[invocation].u4.xy  = subgroupQuadSwapHorizontal(data[1].u4.xy);
    data[invocation].u4.xyz = subgroupQuadSwapHorizontal(data[2].u4.xyz);
    data[invocation].u4     = subgroupQuadSwapHorizontal(data[3].u4);

    data[invocation].d4.x   = subgroupQuadSwapHorizontal(data[0].d4.x);
    data[invocation].d4.xy  = subgroupQuadSwapHorizontal(data[1].d4.xy);
    data[invocation].d4.xyz = subgroupQuadSwapHorizontal(data[2].d4.xyz);
    data[invocation].d4     = subgroupQuadSwapHorizontal(data[3].d4);

    data[invocation].i4.x   =   int(subgroupQuadSwapHorizontal(data[0].i4.x < 0));
    data[invocation].i4.xy  = ivec2(subgroupQuadSwapHorizontal(lessThan(data[1].i4.xy, ivec2(0))));
    data[invocation].i4.xyz = ivec3(subgroupQuadSwapHorizontal(lessThan(data[1].i4.xyz, ivec3(0))));
    data[invocation].i4     = ivec4(subgroupQuadSwapHorizontal(lessThan(data[1].i4, ivec4(0))));

    data[invocation].f4.x   = subgroupQuadSwapVertical(data[0].f4.x);
    data[invocation].f4.xy  = subgroupQuadSwapVertical(data[1].f4.xy);
    data[invocation].f4.xyz = subgroupQuadSwapVertical(data[2].f4.xyz);
    data[invocation].f4     = subgroupQuadSwapVertical(data[3].f4);

    data[invocation].i4.x   = subgroupQuadSwapVertical(data[0].i4.x);
    data[invocation].i4.xy  = subgroupQuadSwapVertical(data[1].i4.xy);
    data[invocation].i4.xyz = subgroupQuadSwapVertical(data[2].i4.xyz);
    data[invocation].i4     = subgroupQuadSwapVertical(data[3].i4);

    data[invocation].u4.x   = subgroupQuadSwapVertical(data[0].u4.x);
    data[invocation].u4.xy  = subgroupQuadSwapVertical(data[1].u4.xy);
    data[invocation].u4.xyz = subgroupQuadSwapVertical(data[2].u4.xyz);
    data[invocation].u4     = subgroupQuadSwapVertical(data[3].u4);

    data[invocation].d4.x   = subgroupQuadSwapVertical(data[0].d4.x);
    data[invocation].d4.xy  = subgroupQuadSwapVertical(data[1].d4.xy);
    data[invocation].d4.xyz = subgroupQuadSwapVertical(data[2].d4.xyz);
    data[invocation].d4     = subgroupQuadSwapVertical(data[3].d4);

    data[invocation].i4.x   =   int(subgroupQuadSwapVertical(data[0].i4.x < 0));
    data[invocation].i4.xy  = ivec2(subgroupQuadSwapVertical(lessThan(data[1].i4.xy, ivec2(0))));
    data[invocation].i4.xyz = ivec3(subgroupQuadSwapVertical(lessThan(data[1].i4.xyz, ivec3(0))));
    data[invocation].i4     = ivec4(subgroupQuadSwapVertical(lessThan(data[1].i4, ivec4(0))));

    data[invocation].f4.x   = subgroupQuadSwapDiagonal(data[0].f4.x);
    data[invocation].f4.xy  = subgroupQuadSwapDiagonal(data[1].f4.xy);
    data[invocation].f4.xyz = subgroupQuadSwapDiagonal(data[2].f4.xyz);
    data[invocation].f4     = subgroupQuadSwapDiagonal(data[3].f4);

    data[invocation].i4.x   = subgroupQuadSwapDiagonal(data[0].i4.x);
    data[invocation].i4.xy  = subgroupQuadSwapDiagonal(data[1].i4.xy);
    data[invocation].i4.xyz = subgroupQuadSwapDiagonal(data[2].i4.xyz);
    data[invocation].i4     = subgroupQuadSwapDiagonal(data[3].i4);

    data[invocation].u4.x   = subgroupQuadSwapDiagonal(data[0].u4.x);
    data[invocation].u4.xy  = subgroupQuadSwapDiagonal(data[1].u4.xy);
    data[invocation].u4.xyz = subgroupQuadSwapDiagonal(data[2].u4.xyz);
    data[invocation].u4     = subgroupQuadSwapDiagonal(data[3].u4);

    data[invocation].d4.x   = subgroupQuadSwapDiagonal(data[0].d4.x);
    data[invocation].d4.xy  = subgroupQuadSwapDiagonal(data[1].d4.xy);
    data[invocation].d4.xyz = subgroupQuadSwapDiagonal(data[2].d4.xyz);
    data[invocation].d4     = subgroupQuadSwapDiagonal(data[3].d4);

    data[invocation].i4.x   =   int(subgroupQuadSwapDiagonal(data[0].i4.x < 0));
    data[invocation].i4.xy  = ivec2(subgroupQuadSwapDiagonal(lessThan(data[1].i4.xy, ivec2(0))));
    data[invocation].i4.xyz = ivec3(subgroupQuadSwapDiagonal(lessThan(data[1].i4.xyz, ivec3(0))));
    data[invocation].i4     = ivec4(subgroupQuadSwapDiagonal(lessThan(data[1].i4, ivec4(0))));
}

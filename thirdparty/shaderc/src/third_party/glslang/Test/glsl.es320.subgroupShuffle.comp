#version 320 es

#extension GL_KHR_shader_subgroup_shuffle: enable

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

    data[0].f4.x   = subgroupShuffle(data[0].f4.x,    invocation);
    data[0].f4.xy  = subgroupShuffle(data[1].f4.xy,   invocation);
    data[0].f4.xyz = subgroupShuffle(data[2].f4.xyz,  invocation);
    data[0].f4     = subgroupShuffle(data[3].f4,      invocation);

    data[0].i4.x   = subgroupShuffle(data[0].i4.x,    invocation);
    data[0].i4.xy  = subgroupShuffle(data[1].i4.xy,   invocation);
    data[0].i4.xyz = subgroupShuffle(data[2].i4.xyz,  invocation);
    data[0].i4     = subgroupShuffle(data[3].i4,      invocation);

    data[1].u4.x   = subgroupShuffle(data[0].u4.x,    invocation);
    data[1].u4.xy  = subgroupShuffle(data[1].u4.xy,   invocation);
    data[1].u4.xyz = subgroupShuffle(data[2].u4.xyz,  invocation);
    data[1].u4     = subgroupShuffle(data[3].u4,      invocation);

    data[1].i4.x   =   int(subgroupShuffle(data[0].i4.x < 0,                   invocation));
    data[1].i4.xy  = ivec2(subgroupShuffle(lessThan(data[1].i4.xy, ivec2(0)),  invocation));
    data[1].i4.xyz = ivec3(subgroupShuffle(lessThan(data[1].i4.xyz, ivec3(0)), invocation));
    data[1].i4     = ivec4(subgroupShuffle(lessThan(data[1].i4, ivec4(0)),     invocation));

    data[2].f4.x   = subgroupShuffleXor(data[0].f4.x,    invocation);
    data[2].f4.xy  = subgroupShuffleXor(data[1].f4.xy,   invocation);
    data[2].f4.xyz = subgroupShuffleXor(data[2].f4.xyz,  invocation);
    data[2].f4     = subgroupShuffleXor(data[3].f4,      invocation);

    data[2].i4.x   = subgroupShuffleXor(data[0].i4.x,    invocation);
    data[2].i4.xy  = subgroupShuffleXor(data[1].i4.xy,   invocation);
    data[2].i4.xyz = subgroupShuffleXor(data[2].i4.xyz,  invocation);
    data[2].i4     = subgroupShuffleXor(data[3].i4,      invocation);

    data[3].u4.x   = subgroupShuffleXor(data[0].u4.x,    invocation);
    data[3].u4.xy  = subgroupShuffleXor(data[1].u4.xy,   invocation);
    data[3].u4.xyz = subgroupShuffleXor(data[2].u4.xyz,  invocation);
    data[3].u4     = subgroupShuffleXor(data[3].u4,      invocation);

    data[3].i4.x   =   int(subgroupShuffleXor(data[0].i4.x < 0,                   invocation));
    data[3].i4.xy  = ivec2(subgroupShuffleXor(lessThan(data[1].i4.xy, ivec2(0)),  invocation));
    data[3].i4.xyz = ivec3(subgroupShuffleXor(lessThan(data[1].i4.xyz, ivec3(0)), invocation));
    data[3].i4     = ivec4(subgroupShuffleXor(lessThan(data[1].i4, ivec4(0)),     invocation));
}

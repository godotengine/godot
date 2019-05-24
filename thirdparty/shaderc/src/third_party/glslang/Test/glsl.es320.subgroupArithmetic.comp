#version 320 es

#extension GL_KHR_shader_subgroup_arithmetic: enable

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

    data[0].f4.x   = subgroupAdd(data[0].f4.x);
    data[0].f4.xy  = subgroupAdd(data[1].f4.xy);
    data[0].f4.xyz = subgroupAdd(data[2].f4.xyz);
    data[0].f4     = subgroupAdd(data[3].f4);

    data[1].i4.x   = subgroupAdd(data[0].i4.x);
    data[1].i4.xy  = subgroupAdd(data[1].i4.xy);
    data[1].i4.xyz = subgroupAdd(data[2].i4.xyz);
    data[1].i4     = subgroupAdd(data[3].i4);

    data[2].u4.x   = subgroupAdd(data[0].u4.x);
    data[2].u4.xy  = subgroupAdd(data[1].u4.xy);
    data[2].u4.xyz = subgroupAdd(data[2].u4.xyz);
    data[2].u4     = subgroupAdd(data[3].u4);

    data[3].f4.x   = subgroupMul(data[0].f4.x);
    data[3].f4.xy  = subgroupMul(data[1].f4.xy);
    data[3].f4.xyz = subgroupMul(data[2].f4.xyz);
    data[3].f4     = subgroupMul(data[3].f4);

    data[0].i4.x   = subgroupMul(data[0].i4.x);
    data[0].i4.xy  = subgroupMul(data[1].i4.xy);
    data[0].i4.xyz = subgroupMul(data[2].i4.xyz);
    data[0].i4     = subgroupMul(data[3].i4);

    data[1].u4.x   = subgroupMul(data[0].u4.x);
    data[1].u4.xy  = subgroupMul(data[1].u4.xy);
    data[1].u4.xyz = subgroupMul(data[2].u4.xyz);
    data[1].u4     = subgroupMul(data[3].u4);

    data[2].f4.x   = subgroupMin(data[0].f4.x);
    data[2].f4.xy  = subgroupMin(data[1].f4.xy);
    data[2].f4.xyz = subgroupMin(data[2].f4.xyz);
    data[2].f4     = subgroupMin(data[3].f4);

    data[3].i4.x   = subgroupMin(data[0].i4.x);
    data[3].i4.xy  = subgroupMin(data[1].i4.xy);
    data[3].i4.xyz = subgroupMin(data[2].i4.xyz);
    data[3].i4     = subgroupMin(data[3].i4);

    data[0].u4.x   = subgroupMin(data[0].u4.x);
    data[0].u4.xy  = subgroupMin(data[1].u4.xy);
    data[0].u4.xyz = subgroupMin(data[2].u4.xyz);
    data[0].u4     = subgroupMin(data[3].u4);

    data[1].f4.x   = subgroupMax(data[0].f4.x);
    data[1].f4.xy  = subgroupMax(data[1].f4.xy);
    data[1].f4.xyz = subgroupMax(data[2].f4.xyz);
    data[1].f4     = subgroupMax(data[3].f4);

    data[2].i4.x   = subgroupMax(data[0].i4.x);
    data[2].i4.xy  = subgroupMax(data[1].i4.xy);
    data[2].i4.xyz = subgroupMax(data[2].i4.xyz);
    data[2].i4     = subgroupMax(data[3].i4);

    data[3].u4.x   = subgroupMax(data[0].u4.x);
    data[3].u4.xy  = subgroupMax(data[1].u4.xy);
    data[3].u4.xyz = subgroupMax(data[2].u4.xyz);
    data[3].u4     = subgroupMax(data[3].u4);

    data[0].i4.x   = subgroupAnd(data[0].i4.x);
    data[0].i4.xy  = subgroupAnd(data[1].i4.xy);
    data[0].i4.xyz = subgroupAnd(data[2].i4.xyz);
    data[0].i4     = subgroupAnd(data[3].i4);

    data[1].u4.x   = subgroupAnd(data[0].u4.x);
    data[1].u4.xy  = subgroupAnd(data[1].u4.xy);
    data[1].u4.xyz = subgroupAnd(data[2].u4.xyz);
    data[1].u4     = subgroupAnd(data[3].u4);

    data[2].i4.x   =   int(subgroupAnd(data[0].i4.x < 0));
    data[2].i4.xy  = ivec2(subgroupAnd(lessThan(data[1].i4.xy, ivec2(0))));
    data[2].i4.xyz = ivec3(subgroupAnd(lessThan(data[1].i4.xyz, ivec3(0))));
    data[2].i4     = ivec4(subgroupAnd(lessThan(data[1].i4, ivec4(0))));

    data[3].i4.x   = subgroupOr(data[0].i4.x);
    data[3].i4.xy  = subgroupOr(data[1].i4.xy);
    data[3].i4.xyz = subgroupOr(data[2].i4.xyz);
    data[3].i4     = subgroupOr(data[3].i4);

    data[0].u4.x   = subgroupOr(data[0].u4.x);
    data[0].u4.xy  = subgroupOr(data[1].u4.xy);
    data[0].u4.xyz = subgroupOr(data[2].u4.xyz);
    data[0].u4     = subgroupOr(data[3].u4);

    data[1].i4.x   =   int(subgroupOr(data[0].i4.x < 0));
    data[1].i4.xy  = ivec2(subgroupOr(lessThan(data[1].i4.xy, ivec2(0))));
    data[1].i4.xyz = ivec3(subgroupOr(lessThan(data[1].i4.xyz, ivec3(0))));
    data[1].i4     = ivec4(subgroupOr(lessThan(data[1].i4, ivec4(0))));

    data[2].i4.x   = subgroupXor(data[0].i4.x);
    data[2].i4.xy  = subgroupXor(data[1].i4.xy);
    data[2].i4.xyz = subgroupXor(data[2].i4.xyz);
    data[2].i4     = subgroupXor(data[3].i4);

    data[3].u4.x   = subgroupXor(data[0].u4.x);
    data[3].u4.xy  = subgroupXor(data[1].u4.xy);
    data[3].u4.xyz = subgroupXor(data[2].u4.xyz);
    data[3].u4     = subgroupXor(data[3].u4);

    data[0].i4.x   =   int(subgroupXor(data[0].i4.x < 0));
    data[0].i4.xy  = ivec2(subgroupXor(lessThan(data[1].i4.xy, ivec2(0))));
    data[0].i4.xyz = ivec3(subgroupXor(lessThan(data[1].i4.xyz, ivec3(0))));
    data[0].i4     = ivec4(subgroupXor(lessThan(data[1].i4, ivec4(0))));

    data[1].f4.x   = subgroupInclusiveAdd(data[0].f4.x);
    data[1].f4.xy  = subgroupInclusiveAdd(data[1].f4.xy);
    data[1].f4.xyz = subgroupInclusiveAdd(data[2].f4.xyz);
    data[1].f4     = subgroupInclusiveAdd(data[3].f4);

    data[2].i4.x   = subgroupInclusiveAdd(data[0].i4.x);
    data[2].i4.xy  = subgroupInclusiveAdd(data[1].i4.xy);
    data[2].i4.xyz = subgroupInclusiveAdd(data[2].i4.xyz);
    data[2].i4     = subgroupInclusiveAdd(data[3].i4);

    data[3].u4.x   = subgroupInclusiveAdd(data[0].u4.x);
    data[3].u4.xy  = subgroupInclusiveAdd(data[1].u4.xy);
    data[3].u4.xyz = subgroupInclusiveAdd(data[2].u4.xyz);
    data[3].u4     = subgroupInclusiveAdd(data[3].u4);

    data[0].f4.x   = subgroupInclusiveMul(data[0].f4.x);
    data[0].f4.xy  = subgroupInclusiveMul(data[1].f4.xy);
    data[0].f4.xyz = subgroupInclusiveMul(data[2].f4.xyz);
    data[0].f4     = subgroupInclusiveMul(data[3].f4);

    data[1].i4.x   = subgroupInclusiveMul(data[0].i4.x);
    data[1].i4.xy  = subgroupInclusiveMul(data[1].i4.xy);
    data[1].i4.xyz = subgroupInclusiveMul(data[2].i4.xyz);
    data[1].i4     = subgroupInclusiveMul(data[3].i4);

    data[2].u4.x   = subgroupInclusiveMul(data[0].u4.x);
    data[2].u4.xy  = subgroupInclusiveMul(data[1].u4.xy);
    data[2].u4.xyz = subgroupInclusiveMul(data[2].u4.xyz);
    data[2].u4     = subgroupInclusiveMul(data[3].u4);

    data[3].f4.x   = subgroupInclusiveMin(data[0].f4.x);
    data[3].f4.xy  = subgroupInclusiveMin(data[1].f4.xy);
    data[3].f4.xyz = subgroupInclusiveMin(data[2].f4.xyz);
    data[3].f4     = subgroupInclusiveMin(data[3].f4);

    data[0].i4.x   = subgroupInclusiveMin(data[0].i4.x);
    data[0].i4.xy  = subgroupInclusiveMin(data[1].i4.xy);
    data[0].i4.xyz = subgroupInclusiveMin(data[2].i4.xyz);
    data[0].i4     = subgroupInclusiveMin(data[3].i4);

    data[1].u4.x   = subgroupInclusiveMin(data[0].u4.x);
    data[1].u4.xy  = subgroupInclusiveMin(data[1].u4.xy);
    data[1].u4.xyz = subgroupInclusiveMin(data[2].u4.xyz);
    data[1].u4     = subgroupInclusiveMin(data[3].u4);

    data[2].f4.x   = subgroupInclusiveMax(data[0].f4.x);
    data[2].f4.xy  = subgroupInclusiveMax(data[1].f4.xy);
    data[2].f4.xyz = subgroupInclusiveMax(data[2].f4.xyz);
    data[2].f4     = subgroupInclusiveMax(data[3].f4);

    data[3].i4.x   = subgroupInclusiveMax(data[0].i4.x);
    data[3].i4.xy  = subgroupInclusiveMax(data[1].i4.xy);
    data[3].i4.xyz = subgroupInclusiveMax(data[2].i4.xyz);
    data[3].i4     = subgroupInclusiveMax(data[3].i4);

    data[0].u4.x   = subgroupInclusiveMax(data[0].u4.x);
    data[0].u4.xy  = subgroupInclusiveMax(data[1].u4.xy);
    data[0].u4.xyz = subgroupInclusiveMax(data[2].u4.xyz);
    data[0].u4     = subgroupInclusiveMax(data[3].u4);

    data[1].i4.x   = subgroupInclusiveAnd(data[0].i4.x);
    data[1].i4.xy  = subgroupInclusiveAnd(data[1].i4.xy);
    data[1].i4.xyz = subgroupInclusiveAnd(data[2].i4.xyz);
    data[1].i4     = subgroupInclusiveAnd(data[3].i4);

    data[2].u4.x   = subgroupInclusiveAnd(data[0].u4.x);
    data[2].u4.xy  = subgroupInclusiveAnd(data[1].u4.xy);
    data[2].u4.xyz = subgroupInclusiveAnd(data[2].u4.xyz);
    data[2].u4     = subgroupInclusiveAnd(data[3].u4);

    data[3].i4.x   =   int(subgroupInclusiveAnd(data[0].i4.x < 0));
    data[3].i4.xy  = ivec2(subgroupInclusiveAnd(lessThan(data[1].i4.xy, ivec2(0))));
    data[3].i4.xyz = ivec3(subgroupInclusiveAnd(lessThan(data[1].i4.xyz, ivec3(0))));
    data[3].i4     = ivec4(subgroupInclusiveAnd(lessThan(data[1].i4, ivec4(0))));

    data[0].i4.x   = subgroupInclusiveOr(data[0].i4.x);
    data[0].i4.xy  = subgroupInclusiveOr(data[1].i4.xy);
    data[0].i4.xyz = subgroupInclusiveOr(data[2].i4.xyz);
    data[0].i4     = subgroupInclusiveOr(data[3].i4);

    data[1].u4.x   = subgroupInclusiveOr(data[0].u4.x);
    data[1].u4.xy  = subgroupInclusiveOr(data[1].u4.xy);
    data[1].u4.xyz = subgroupInclusiveOr(data[2].u4.xyz);
    data[1].u4     = subgroupInclusiveOr(data[3].u4);

    data[2].i4.x   =   int(subgroupInclusiveOr(data[0].i4.x < 0));
    data[2].i4.xy  = ivec2(subgroupInclusiveOr(lessThan(data[1].i4.xy, ivec2(0))));
    data[2].i4.xyz = ivec3(subgroupInclusiveOr(lessThan(data[1].i4.xyz, ivec3(0))));
    data[2].i4     = ivec4(subgroupInclusiveOr(lessThan(data[1].i4, ivec4(0))));

    data[3].i4.x   = subgroupInclusiveXor(data[0].i4.x);
    data[3].i4.xy  = subgroupInclusiveXor(data[1].i4.xy);
    data[3].i4.xyz = subgroupInclusiveXor(data[2].i4.xyz);
    data[3].i4     = subgroupInclusiveXor(data[3].i4);

    data[0].u4.x   = subgroupInclusiveXor(data[0].u4.x);
    data[0].u4.xy  = subgroupInclusiveXor(data[1].u4.xy);
    data[0].u4.xyz = subgroupInclusiveXor(data[2].u4.xyz);
    data[0].u4     = subgroupInclusiveXor(data[3].u4);

    data[1].i4.x   =   int(subgroupInclusiveXor(data[0].i4.x < 0));
    data[1].i4.xy  = ivec2(subgroupInclusiveXor(lessThan(data[1].i4.xy, ivec2(0))));
    data[1].i4.xyz = ivec3(subgroupInclusiveXor(lessThan(data[1].i4.xyz, ivec3(0))));
    data[1].i4     = ivec4(subgroupInclusiveXor(lessThan(data[1].i4, ivec4(0))));

    data[2].f4.x   = subgroupExclusiveAdd(data[0].f4.x);
    data[2].f4.xy  = subgroupExclusiveAdd(data[1].f4.xy);
    data[2].f4.xyz = subgroupExclusiveAdd(data[2].f4.xyz);
    data[2].f4     = subgroupExclusiveAdd(data[3].f4);

    data[3].i4.x   = subgroupExclusiveAdd(data[0].i4.x);
    data[3].i4.xy  = subgroupExclusiveAdd(data[1].i4.xy);
    data[3].i4.xyz = subgroupExclusiveAdd(data[2].i4.xyz);
    data[3].i4     = subgroupExclusiveAdd(data[3].i4);

    data[0].u4.x   = subgroupExclusiveAdd(data[0].u4.x);
    data[0].u4.xy  = subgroupExclusiveAdd(data[1].u4.xy);
    data[0].u4.xyz = subgroupExclusiveAdd(data[2].u4.xyz);
    data[0].u4     = subgroupExclusiveAdd(data[3].u4);

    data[1].f4.x   = subgroupExclusiveMul(data[0].f4.x);
    data[1].f4.xy  = subgroupExclusiveMul(data[1].f4.xy);
    data[1].f4.xyz = subgroupExclusiveMul(data[2].f4.xyz);
    data[1].f4     = subgroupExclusiveMul(data[3].f4);

    data[2].i4.x   = subgroupExclusiveMul(data[0].i4.x);
    data[2].i4.xy  = subgroupExclusiveMul(data[1].i4.xy);
    data[2].i4.xyz = subgroupExclusiveMul(data[2].i4.xyz);
    data[2].i4     = subgroupExclusiveMul(data[3].i4);

    data[3].u4.x   = subgroupExclusiveMul(data[0].u4.x);
    data[3].u4.xy  = subgroupExclusiveMul(data[1].u4.xy);
    data[3].u4.xyz = subgroupExclusiveMul(data[2].u4.xyz);
    data[3].u4     = subgroupExclusiveMul(data[3].u4);

    data[0].f4.x   = subgroupExclusiveMin(data[0].f4.x);
    data[0].f4.xy  = subgroupExclusiveMin(data[1].f4.xy);
    data[0].f4.xyz = subgroupExclusiveMin(data[2].f4.xyz);
    data[0].f4     = subgroupExclusiveMin(data[3].f4);

    data[1].i4.x   = subgroupExclusiveMin(data[0].i4.x);
    data[1].i4.xy  = subgroupExclusiveMin(data[1].i4.xy);
    data[1].i4.xyz = subgroupExclusiveMin(data[2].i4.xyz);
    data[1].i4     = subgroupExclusiveMin(data[3].i4);

    data[2].u4.x   = subgroupExclusiveMin(data[0].u4.x);
    data[2].u4.xy  = subgroupExclusiveMin(data[1].u4.xy);
    data[2].u4.xyz = subgroupExclusiveMin(data[2].u4.xyz);
    data[2].u4     = subgroupExclusiveMin(data[3].u4);

    data[3].f4.x   = subgroupExclusiveMax(data[0].f4.x);
    data[3].f4.xy  = subgroupExclusiveMax(data[1].f4.xy);
    data[3].f4.xyz = subgroupExclusiveMax(data[2].f4.xyz);
    data[3].f4     = subgroupExclusiveMax(data[3].f4);

    data[0].i4.x   = subgroupExclusiveMax(data[0].i4.x);
    data[0].i4.xy  = subgroupExclusiveMax(data[1].i4.xy);
    data[0].i4.xyz = subgroupExclusiveMax(data[2].i4.xyz);
    data[0].i4     = subgroupExclusiveMax(data[3].i4);

    data[1].u4.x   = subgroupExclusiveMax(data[0].u4.x);
    data[1].u4.xy  = subgroupExclusiveMax(data[1].u4.xy);
    data[1].u4.xyz = subgroupExclusiveMax(data[2].u4.xyz);
    data[1].u4     = subgroupExclusiveMax(data[3].u4);

    data[2].i4.x   = subgroupExclusiveAnd(data[0].i4.x);
    data[2].i4.xy  = subgroupExclusiveAnd(data[1].i4.xy);
    data[2].i4.xyz = subgroupExclusiveAnd(data[2].i4.xyz);
    data[2].i4     = subgroupExclusiveAnd(data[3].i4);

    data[3].u4.x   = subgroupExclusiveAnd(data[0].u4.x);
    data[3].u4.xy  = subgroupExclusiveAnd(data[1].u4.xy);
    data[3].u4.xyz = subgroupExclusiveAnd(data[2].u4.xyz);
    data[3].u4     = subgroupExclusiveAnd(data[3].u4);

    data[0].i4.x   =   int(subgroupExclusiveAnd(data[0].i4.x < 0));
    data[0].i4.xy  = ivec2(subgroupExclusiveAnd(lessThan(data[1].i4.xy, ivec2(0))));
    data[0].i4.xyz = ivec3(subgroupExclusiveAnd(lessThan(data[1].i4.xyz, ivec3(0))));
    data[0].i4     = ivec4(subgroupExclusiveAnd(lessThan(data[1].i4, ivec4(0))));

    data[1].i4.x   = subgroupExclusiveOr(data[0].i4.x);
    data[1].i4.xy  = subgroupExclusiveOr(data[1].i4.xy);
    data[1].i4.xyz = subgroupExclusiveOr(data[2].i4.xyz);
    data[1].i4     = subgroupExclusiveOr(data[3].i4);

    data[2].u4.x   = subgroupExclusiveOr(data[0].u4.x);
    data[2].u4.xy  = subgroupExclusiveOr(data[1].u4.xy);
    data[2].u4.xyz = subgroupExclusiveOr(data[2].u4.xyz);
    data[2].u4     = subgroupExclusiveOr(data[3].u4);

    data[3].i4.x   =   int(subgroupExclusiveOr(data[0].i4.x < 0));
    data[3].i4.xy  = ivec2(subgroupExclusiveOr(lessThan(data[1].i4.xy, ivec2(0))));
    data[3].i4.xyz = ivec3(subgroupExclusiveOr(lessThan(data[1].i4.xyz, ivec3(0))));
    data[3].i4     = ivec4(subgroupExclusiveOr(lessThan(data[1].i4, ivec4(0))));

    data[0].i4.x   = subgroupExclusiveXor(data[0].i4.x);
    data[0].i4.xy  = subgroupExclusiveXor(data[1].i4.xy);
    data[0].i4.xyz = subgroupExclusiveXor(data[2].i4.xyz);
    data[0].i4     = subgroupExclusiveXor(data[3].i4);

    data[1].u4.x   = subgroupExclusiveXor(data[0].u4.x);
    data[1].u4.xy  = subgroupExclusiveXor(data[1].u4.xy);
    data[1].u4.xyz = subgroupExclusiveXor(data[2].u4.xyz);
    data[1].u4     = subgroupExclusiveXor(data[3].u4);

    data[2].i4.x   =   int(subgroupExclusiveXor(data[0].i4.x < 0));
    data[2].i4.xy  = ivec2(subgroupExclusiveXor(lessThan(data[1].i4.xy, ivec2(0))));
    data[2].i4.xyz = ivec3(subgroupExclusiveXor(lessThan(data[1].i4.xyz, ivec3(0))));
    data[2].i4     = ivec4(subgroupExclusiveXor(lessThan(data[1].i4, ivec4(0))));
}

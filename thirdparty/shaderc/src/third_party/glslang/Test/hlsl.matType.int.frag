
void TestIntMatTypes()
{
    int1x1 i1x1;
    int2x1 i2x1;
    int3x1 i3x1;
    int4x1 i4x1;

    int1x2 i1x2;
    int2x2 i2x2;
    int3x2 i3x2;
    int4x2 i4x2;

    int1x3 i1x3;
    int2x3 i2x3;
    int3x3 i3x3;
    int4x3 i4x3;

    int1x4 i1x4;
    int2x4 i2x4;
    int3x4 i3x4;
    int4x4 i4x4;

    // TODO: Currently SPIR-V disallows Nx1 or 1xN mats.
    int1x1 r00 = transpose(i1x1);
    int1x2 r01 = transpose(i2x1);
    int1x3 r02 = transpose(i3x1);
    int1x4 r03 = transpose(i4x1);

    int2x1 r10 = transpose(i1x2);
    int2x2 r11 = transpose(i2x2);
    int2x3 r12 = transpose(i3x2);
    int2x4 r13 = transpose(i4x2);

    int3x1 r20 = transpose(i1x3);
    int3x2 r21 = transpose(i2x3);
    int3x3 r22 = transpose(i3x3);
    int3x4 r23 = transpose(i4x3);

    int4x1 r30 = transpose(i1x4);
    int4x2 r31 = transpose(i2x4);
    int4x3 r32 = transpose(i3x4);
    int4x4 r33 = transpose(i4x4);
}

void TestUintMatTypes()
{
    uint1x1 u1x1;
    uint2x1 u2x1;
    uint3x1 u3x1;
    uint4x1 u4x1;
    
    uint1x2 u1x2;
    uint2x2 u2x2;
    uint3x2 u3x2;
    uint4x2 u4x2;
    
    uint1x3 u1x3;
    uint2x3 u2x3;
    uint3x3 u3x3;
    uint4x3 u4x3;
    
    uint1x4 u1x4;
    uint2x4 u2x4;
    uint3x4 u3x4;
    uint4x4 u4x4;
    
    // TODO: Currently SPIR-V disallows Nx1 or 1xN mats.
    uint1x1 r00 = transpose(u1x1);
    uint1x2 r01 = transpose(u2x1);
    uint1x3 r02 = transpose(u3x1);
    uint1x4 r03 = transpose(u4x1);
    
    uint2x1 r10 = transpose(u1x2);
    uint2x2 r11 = transpose(u2x2);
    uint2x3 r12 = transpose(u3x2);
    uint2x4 r13 = transpose(u4x2);
    
    uint3x1 r20 = transpose(u1x3);
    uint3x2 r21 = transpose(u2x3);
    uint3x3 r22 = transpose(u3x3);
    uint3x4 r23 = transpose(u4x3);
    
    uint4x1 r30 = transpose(u1x4);
    uint4x2 r31 = transpose(u2x4);
    uint4x3 r32 = transpose(u3x4);
    uint4x4 r33 = transpose(u4x4);
}

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = float4(0,0,0,0);
    return ps_output;
};

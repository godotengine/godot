
void TestMatNx1()
{
    float1x1 f1x1;
    float2x1 f2x1;
    float3x1 f3x1;
    float4x1 f4x1;

    float1x2 f1x2;
    float1x3 f1x3;
    float1x4 f1x4;

    float1x1 r00 = transpose(f1x1);
    float1x2 r01 = transpose(f2x1);
    float1x3 r02 = transpose(f3x1);
    float1x4 r03 = transpose(f4x1);

    float1x1 r10 = transpose(f1x1);
    float2x1 r11 = transpose(f1x2);
    float3x1 r12 = transpose(f1x3);
    float4x1 r13 = transpose(f1x4);
}

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
};

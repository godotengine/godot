
void TestBoolMatTypes()
{
    bool1x1 b1x1;
    bool2x1 b2x1;
    bool3x1 b3x1;
    bool4x1 b4x1;
    
    bool1x2 b1x2;
    bool2x2 b2x2;
    bool3x2 b3x2;
    bool4x2 b4x2;
    
    bool1x3 b1x3;
    bool2x3 b2x3;
    bool3x3 b3x3;
    bool4x3 b4x3;
    
    bool1x4 b1x4;
    bool2x4 b2x4;
    bool3x4 b3x4;
    bool4x4 b4x4;
    
    // TODO: Currently SPIR-V disallows Nx1 or 1xN mats.
    bool1x1 r00 = transpose(b1x1);
    bool1x2 r01 = transpose(b2x1);
    bool1x3 r02 = transpose(b3x1);
    bool1x4 r03 = transpose(b4x1);
    
    bool2x1 r10 = transpose(b1x2);
    bool2x2 r11 = transpose(b2x2);
    bool2x3 r12 = transpose(b3x2);
    bool2x4 r13 = transpose(b4x2);
    
    bool3x1 r20 = transpose(b1x3);
    bool3x2 r21 = transpose(b2x3);
    bool3x3 r22 = transpose(b3x3);
    bool3x4 r23 = transpose(b4x3);
    
    bool4x1 r30 = transpose(b1x4);
    bool4x2 r31 = transpose(b2x4);
    bool4x3 r32 = transpose(b3x4);
    bool4x4 r33 = transpose(b4x4);
}

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = float4(0,0,0,0);
    return ps_output;
};

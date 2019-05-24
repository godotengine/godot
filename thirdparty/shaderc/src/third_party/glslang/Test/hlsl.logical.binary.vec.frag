struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform bool4 b4a, b4b;
uniform bool  b1a, b1b;

PS_OUTPUT main()
{
    bool4 r00 = !b4a;
    bool4 r01 = b4a && b4b;  // vec, vec
    bool4 r02 = b4a || b4b;  // vec, vec

    bool4 r10 = b1a && b4b;  // scalar, vec
    bool4 r11 = b1a || b4b;  // scalar, vec

    bool4 r20 = b4a && b1b;  // vec, scalar
    bool4 r21 = b4a || b1b;  // vec, scalar

    PS_OUTPUT psout;
    psout.Color = r00 || r01 || r02 || r10 || r11 || r20 || r21;
    return psout;
}

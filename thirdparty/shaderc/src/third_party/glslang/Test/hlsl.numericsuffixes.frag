
struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    // Test numeric suffixes
    float r00 = 1.0f;    // float
    uint  r01 = 1u;      // lower uint
    uint  r02 = 2U;      // upper uint
    uint  r03 = 0xabcu;  // lower hex uint
    uint  r04 = 0xABCU;  // upper hex uint (upper 0X is not accepted)
    int   r05 = 5l;      // lower long int
    int   r06 = 6L;      // upper long int
    int   r07 = 071;     // octal
    uint  r08 = 072u;    // unsigned octal
    float r09 = 1.h;     // half
    float r10 = 1.H;     // half
    float r11 = 1.1h;    // half
    float r12 = 1.1H;    // half

    PS_OUTPUT ps_output;
    ps_output.color = r07; // gets 71 octal = 57 decimal
    return ps_output;
}

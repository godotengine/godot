
struct PS_OUTPUT { float4 color : SV_Target0; };

int    i;
uint   u;
float  f;
bool   b;

int2   i2;
uint2  u2;
float2 f2;
bool2  b2;

PS_OUTPUT main()
{
    uint r00  = countbits(f);
    uint2 r01 = reversebits(f2);

    PS_OUTPUT ps_output;
    ps_output.color = float4(0,0,0,0);
    return ps_output;
};

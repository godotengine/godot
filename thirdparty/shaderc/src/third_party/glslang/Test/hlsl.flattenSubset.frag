struct S0
{
    int x;
    int y;
    SamplerState ss;
};

struct S1
{
    float b;
    SamplerState samplerState;
    S0 s0;
    int a;
};

struct S2
{
    int a1;
    int a2;
    int a3;
    int a4;
    int a5;
    S1 resources;
};

SamplerState samp;
Texture2D tex;

float4 main(float4 vpos : VPOS) : COLOR0
{
    S1 s1;
    S2 s2;
    s1.s0.ss = samp;
    s2.resources = s1;
    return tex.Sample(s2.resources.s0.ss, float2(0.5));
}

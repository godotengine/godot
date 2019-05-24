
SamplerState       g_sSamp1 : register(s0);
SamplerState       g_sSamp2;
SamplerState       g_sSamp3[2] : register(s2);
SamplerState       g_sSamp4[3];
SamplerState       g_sSamp5;

SamplerState       g_sSamp_unused1;
SamplerState       g_sSamp_unused2;

Texture1D          g_tTex1 : register(t1);
const uniform Texture1D g_tTex2;
Texture1D          g_tTex3[2] : register(t3);
Texture1D          g_tTex4[3];
Texture1D          g_tTex5;

Texture1D          g_tTex_unused1 : register(t0);
Texture1D          g_tTex_unused2 : register(t2);
Texture1D          g_tTex_unused3;

struct MyStruct_t {
    int a;
    float b;
    float3 c;
};

uniform MyStruct_t mystruct : register(b4);

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform float4 myfloat4_a;
uniform float4 myfloat4_b;
uniform int4 myint4_a;

float4 Func1()
{
    return
        g_tTex1    . Sample(g_sSamp1,    0.1) +
        g_tTex2    . Sample(g_sSamp2,    0.2) +
        g_tTex3[0] . Sample(g_sSamp3[0], 0.3) +
        g_tTex3[1] . Sample(g_sSamp3[1], 0.3) +
        g_tTex4[1] . Sample(g_sSamp4[1], 0.4) +
        g_tTex4[2] . Sample(g_sSamp4[2], 0.4) +
        g_tTex5    . Sample(g_sSamp5,    0.5) +
        mystruct.c[1];
}

float4 Func2()
{
    return
        g_tTex1    . Sample(g_sSamp1,    0.1) +
        g_tTex3[1] . Sample(g_sSamp3[1], 0.3);
}

// Not called from entry point:
float4 Func2_unused()
{
    return
        g_tTex_unused1 . Sample(g_sSamp_unused1, 1.1) +
        g_tTex_unused2 . Sample(g_sSamp_unused2, 1.2);
}

PS_OUTPUT main_ep()
{
    PS_OUTPUT psout;
    psout.Color = Func1() + Func2();
    return psout;
}

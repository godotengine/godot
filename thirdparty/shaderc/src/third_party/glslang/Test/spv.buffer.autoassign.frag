
cbuffer MyUB1 : register(b5)  // explicitly assigned & offsetted
{
    float g_a;
    int g_b;
};

cbuffer MyUB2  // implicitly assigned
{
    float g_c;
};

cbuffer MyUB3  // implicitly assigned
{
    float g_d;
};

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
    PS_OUTPUT psout;
    psout.Color = g_a + g_b + g_c + g_d;
    return psout;
}

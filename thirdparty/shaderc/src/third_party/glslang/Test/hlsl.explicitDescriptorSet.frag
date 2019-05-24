SamplerState       g_sSamp : register(s1);

Texture1D <float4> g_tTex1df4 : register(t0);

SamplerState       g_sSamp2_amb;
uniform float      floatval_amb;

Buffer<float>      floatbuff;

float4 main() : SV_Target0
{
    g_sSamp2_amb;

    return 0;
}

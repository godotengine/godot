SamplerState g_sSamp : register(s0);
SamplerComparisonState g_sSampCmp : register(s1);

uniform Texture1D <float4> g_tTex : register(t3);

float4 main() : SV_Target0
{
    // This texture is used with both shadow modes.  It will need post-compilation
    // legalization.
    g_tTex.SampleCmp(g_sSampCmp, 0.1, 0.75);
    g_tTex.Sample(g_sSamp, 0.1);

    return 0;
}

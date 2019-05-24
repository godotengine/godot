
Texture2D g_nonShadowTex;
Texture2D g_shadowTex;
SamplerState g_shadowSampler;
SamplerComparisonState g_shadowSamplerComp;

float4 main() : SV_Target0
{
    g_shadowTex.SampleCmp(g_shadowSamplerComp, float2(0,0), 0); // OK
    g_nonShadowTex.SampleCmp(g_shadowSampler, float2(0,0), 0);     // ERROR (should be comparison sampler)

    return 0;
}

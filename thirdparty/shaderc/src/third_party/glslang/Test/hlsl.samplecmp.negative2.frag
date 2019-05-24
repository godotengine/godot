
Texture2D g_shadowTex;
SamplerState g_shadowSampler;

float4 main() : SV_Target0
{
    g_shadowTex.GatherCmpRed(g_shadowSampler, float2(0,0), 0, int2(0,0));  // ERROR (should be comparison sampler)

    return 0;
}

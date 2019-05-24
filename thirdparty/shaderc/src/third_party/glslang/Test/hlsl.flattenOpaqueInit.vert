struct FxaaTex { SamplerState smpl; Texture2D tex; };
SamplerState g_tInputTexture_sampler; Texture2D g_tInputTexture;

float4 lookUp(FxaaTex tex)
{
    return tex.tex.Sample(tex.smpl, float2(0.3, 0.4));
}

FxaaTex fillOpaque()
{
    FxaaTex t;
    t.smpl = g_tInputTexture_sampler;
    t.tex = g_tInputTexture;
    return t;
}

float4 main() : SV_TARGET0
{
    FxaaTex tex1 = { g_tInputTexture_sampler, g_tInputTexture };
    float4 res = lookUp(tex1);
    FxaaTex tex2 = fillOpaque();
    res += lookUp(tex2);
    FxaaTex tex3 = tex1;
    res += lookUp(tex3);

    return res;
}

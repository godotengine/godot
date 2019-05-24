struct FxaaTex { SamplerState smpl; Texture2D tex; float f; };
SamplerState g_tInputTexture_sampler; Texture2D g_tInputTexture;

float4 lookUp(FxaaTex tex)
{
    return tex.tex.Sample(tex.smpl, float2(tex.f, tex.f));
}

float4 main() : SV_TARGET0
{
    FxaaTex tex = { g_tInputTexture_sampler, g_tInputTexture, 0.5 };
    return lookUp(tex);
}
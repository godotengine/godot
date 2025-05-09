Texture2D theTexture : register(t0);
SamplerState theSampler : register(s0);

#include "D3D11_PixelShader_Common.hlsli"

float4 main(PixelShaderInput input) : SV_TARGET
{
    return GetOutputColor(theTexture.Sample(theSampler, input.tex)) * input.color;
}

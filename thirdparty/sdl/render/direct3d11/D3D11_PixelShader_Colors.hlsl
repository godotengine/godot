
#include "D3D11_PixelShader_Common.hlsli"

float4 main(PixelShaderInput input) : SV_TARGET
{
    return GetOutputColor(1.0) * input.color;
}

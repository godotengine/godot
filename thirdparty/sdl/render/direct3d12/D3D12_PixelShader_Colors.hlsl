
#include "D3D12_PixelShader_Common.hlsli"

[RootSignature(ColorRS)]
float4 main(PixelShaderInput input) : SV_TARGET0
{
    return GetOutputColor(1.0) * input.color;
}

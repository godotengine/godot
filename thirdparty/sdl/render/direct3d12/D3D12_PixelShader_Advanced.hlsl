
#include "D3D12_PixelShader_Common.hlsli"

[RootSignature(AdvancedRS)]
float4 main(PixelShaderInput input) : SV_TARGET
{
    return AdvancedPixelShader(input);
}

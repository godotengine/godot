
#include "VULKAN_PixelShader_Common.hlsli"

float4 main(PixelShaderInput input) : SV_TARGET
{
	return AdvancedPixelShader(input);
}

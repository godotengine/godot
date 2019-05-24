float PixelShaderFunctionS(uint inF0)
{
    return f16tof32(inF0);
}

float1 PixelShaderFunction1(uint1 inF0)
{
    return f16tof32(inF0);
}

float2 PixelShaderFunction2(uint2 inF0)
{
    return f16tof32(inF0);
}

float3 PixelShaderFunction3(uint3 inF0)
{
    return f16tof32(inF0);
}

float4 PixelShaderFunction(uint4 inF0)
{
    return f16tof32(inF0);
}

float4 main() : SV_Target0
{
    return 0;
}

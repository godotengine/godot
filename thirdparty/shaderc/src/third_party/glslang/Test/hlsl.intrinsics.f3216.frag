uint PixelShaderFunctionS(float inF0)
{
    return f32tof16(inF0);
}

uint1 PixelShaderFunction1(float1 inF0)
{
    return f32tof16(inF0);
}

uint2 PixelShaderFunction2(float2 inF0)
{
    return f32tof16(inF0);
}

uint3 PixelShaderFunction3(float3 inF0)
{
    return f32tof16(inF0);
}

uint4 PixelShaderFunction(float4 inF0)
{
    return f32tof16(inF0);
}

float4 main() : SV_Target0
{
    return 0;
}

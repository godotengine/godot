float4 PixelShaderFunction(float4 input) : COLOR0
{
    return (float4)input + (int4)input + (float4)1.198;
}

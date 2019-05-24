float4 PixelShaderFunction(float4 input1, float4 input2) : COLOR0
{
    return max(input1, input2);
}

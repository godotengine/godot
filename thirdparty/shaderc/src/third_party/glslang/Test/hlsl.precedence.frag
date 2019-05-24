float4 PixelShaderFunction(
    float4 a1,
    float4 a2,
    float4 a3,
    float4 a4
    ) : COLOR0
{
    return a1 + a2 * a3 + a4 + float4(a1.rgb * a2.rgb, a3.a);
}

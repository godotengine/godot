
float PixelShaderFunctionS(float inF0, float inF1)
{
    float r000 = frexp(inF0, inF1);
    return 0.0;
}

float2 PixelShaderFunction2(float2 inF0, float2 inF1)
{
    float2 r000 = frexp(inF0, inF1);
    return float2(1,2);
}

float3 PixelShaderFunction3(float3 inF0, float3 inF1)
{
    float3 r000 = frexp(inF0, inF1);
    return float3(1,2,3);
}

float4 PixelShaderFunction(float4 inF0, float4 inF1)
{
    float4 r000 = frexp(inF0, inF1);
    return float4(1,2,3,4);
}

// TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
#define MATFNS(MT)                          \
    MT r000 = frexp(inF0, inF1);

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
};

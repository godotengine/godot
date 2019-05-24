float VertexShaderFunctionS(float inF0, float inF1)
{
    frexp(inF0, inF1);
    return 0.0;
}

float2 VertexShaderFunction2(float2 inF0, float2 inF1)
{
    frexp(inF0, inF1);
    return float2(1,2);
}

float3 VertexShaderFunction3(float3 inF0, float3 inF1)
{
    frexp(inF0, inF1);
    return float3(1,2,3);
}

float4 VertexShaderFunction4(float4 inF0, float4 inF1)
{
    frexp(inF0, inF1);
    return float4(1,2,3,4);
}

// TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
#define MATFNS() \
    frexp(inF0, inF1);


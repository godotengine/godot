
cbuffer ConstantBuffer : register( b0 )
{
    matrix World;
    matrix View;
    matrix Projection;
};

struct VS_INPUT
{
    float4 Pos : POSITION;
    float3 Norm : NORMAL;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float3 Norm : TEXCOORD0;
};

PS_INPUT main( VS_INPUT input )
{
    int ConstantBuffer = 42;  // test ConstantBuffer as an identifier

    PS_INPUT output = (PS_INPUT)0;
    output.Pos = mul( input.Pos, World );
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( output.Pos, Projection );
    output.Norm = mul( input.Norm, World );  // Work when output.Norm = mul( input.Norm, (float3x3)World );

    return output;
}

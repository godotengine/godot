#pragma pack_matrix( row_major )

cbuffer VertexShaderConstants : register(b0)
{
    matrix model;
    matrix projectionAndView;
};

struct VertexShaderInput
{
    float3 pos : POSITION;
    float2 tex : TEXCOORD0;
    float4 color : COLOR0;
};

struct VertexShaderOutput
{
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
    float4 color : COLOR0;
};

VertexShaderOutput main(VertexShaderInput input)
{
    VertexShaderOutput output;
    float4 pos = float4(input.pos, 1.0f);

    // Transform the vertex position into projected space.
    pos = mul(pos, model);
    pos = mul(pos, projectionAndView);
    output.pos = pos;

    // Pass through texture coordinates and color values without transformation
    output.tex = input.tex;
    output.color = input.color;

    return output;
}

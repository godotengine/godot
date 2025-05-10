#pragma pack_matrix( row_major )

struct VertexShaderConstants
{
    matrix model;
    matrix projectionAndView;
};
[[vk::push_constant]]
ConstantBuffer<VertexShaderConstants> pushConstants;

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
    [[vk::builtin("PointSize")]] float pointSize : SV_PointSize;
};

VertexShaderOutput mainColor(VertexShaderInput input)
{
    VertexShaderOutput output;
    float4 pos = float4(input.pos, 1.0f);

    // Transform the vertex position into projected space.
    pos = mul(pos, pushConstants.model);
    pos = mul(pos, pushConstants.projectionAndView);
    output.pos = pos;

    // Pass through texture coordinates and color values without transformation
    output.tex = input.tex;
    output.color = input.color;

    // Always output pointSize so that this VS can be used with points
    output.pointSize = 1.0;

    return output;
}


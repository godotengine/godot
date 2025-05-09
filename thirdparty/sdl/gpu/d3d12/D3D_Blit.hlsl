#define BlitRS \
    "DescriptorTable ( Sampler(s0, space=2), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "DescriptorTable ( SRV(t0, space=2), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "CBV(b0, space=3, visibility = SHADER_VISIBILITY_PIXEL),"\

struct VertexToPixel
{
    float2 tex : TEXCOORD0;
    float4 pos : SV_POSITION;
};

cbuffer SourceRegionBuffer : register(b0, space3)
{
    float2 UVLeftTop;
    float2 UVDimensions;
    uint MipLevel;
    float LayerOrDepth;
};

Texture2D SourceTexture2D : register(t0, space2);
Texture2DArray SourceTexture2DArray : register(t0, space2);
Texture3D SourceTexture3D : register(t0, space2);
TextureCube SourceTextureCube : register(t0, space2);
TextureCubeArray SourceTextureCubeArray : register(t0, space2);
sampler SourceSampler : register(s0, space2);

[RootSignature(BlitRS)]
VertexToPixel FullscreenVert(uint vI : SV_VERTEXID)
{
    float2 inTex = float2((vI << 1) & 2, vI & 2);
    VertexToPixel Out = (VertexToPixel)0;
    Out.tex = inTex;
    Out.pos = float4(inTex * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return Out;
}

[RootSignature(BlitRS)]
float4 BlitFrom2D(VertexToPixel input) : SV_Target0
{
    float2 newCoord = UVLeftTop + UVDimensions * input.tex;
    return SourceTexture2D.SampleLevel(SourceSampler, newCoord, MipLevel);
}

[RootSignature(BlitRS)]
float4 BlitFrom2DArray(VertexToPixel input) : SV_Target0
{
    float3 newCoord = float3(UVLeftTop + UVDimensions * input.tex, (uint)LayerOrDepth);
    return SourceTexture2DArray.SampleLevel(SourceSampler, newCoord, MipLevel);
}

[RootSignature(BlitRS)]
float4 BlitFrom3D(VertexToPixel input) : SV_Target0
{
    float3 newCoord = float3(UVLeftTop + UVDimensions * input.tex, LayerOrDepth);
    return SourceTexture3D.SampleLevel(SourceSampler, newCoord, MipLevel);
}

[RootSignature(BlitRS)]
float4 BlitFromCube(VertexToPixel input) : SV_Target0
{
    // Thanks, Wikipedia! https://en.wikipedia.org/wiki/Cube_mapping
    float3 newCoord;
    float2 scaledUV = UVLeftTop + UVDimensions * input.tex;
    float u = 2.0 * scaledUV.x - 1.0;
    float v = 2.0 * scaledUV.y - 1.0;
    switch ((uint)LayerOrDepth) {
        case 0: newCoord = float3(1.0, -v, -u); break; // POSITIVE X
        case 1: newCoord = float3(-1.0, -v, u); break; // NEGATIVE X
        case 2: newCoord = float3(u, 1.0, -v); break; // POSITIVE Y
        case 3: newCoord = float3(u, -1.0, v); break; // NEGATIVE Y
        case 4: newCoord = float3(u, -v, 1.0); break; // POSITIVE Z
        case 5: newCoord = float3(-u, -v, -1.0); break; // NEGATIVE Z
        default: newCoord = float3(0, 0, 0); break; // silences warning
    }
    return SourceTextureCube.SampleLevel(SourceSampler, newCoord, MipLevel);
}

[RootSignature(BlitRS)]
float4 BlitFromCubeArray(VertexToPixel input) : SV_Target0
{
    // Thanks, Wikipedia! https://en.wikipedia.org/wiki/Cube_mapping
    float3 newCoord;
    float2 scaledUV = UVLeftTop + UVDimensions * input.tex;
    float u = 2.0 * scaledUV.x - 1.0;
    float v = 2.0 * scaledUV.y - 1.0;
    uint ArrayIndex = (uint)LayerOrDepth / 6;
    switch ((uint)LayerOrDepth % 6) {
        case 0: newCoord = float3(1.0, -v, -u); break; // POSITIVE X
        case 1: newCoord = float3(-1.0, -v, u); break; // NEGATIVE X
        case 2: newCoord = float3(u, 1.0, -v); break; // POSITIVE Y
        case 3: newCoord = float3(u, -1.0, v); break; // NEGATIVE Y
        case 4: newCoord = float3(u, -v, 1.0); break; // POSITIVE Z
        case 5: newCoord = float3(-u, -v, -1.0); break; // NEGATIVE Z
        default: newCoord = float3(0, 0, 0); break; // silences warning
    }
    return SourceTextureCubeArray.SampleLevel(SourceSampler, float4(newCoord, float(ArrayIndex)), MipLevel);
}

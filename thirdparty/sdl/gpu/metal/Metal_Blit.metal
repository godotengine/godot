#include <metal_stdlib>
using namespace metal;

struct VertexToFragment {
    float2 tex;
    float4 pos [[position]];
};

struct SourceRegion {
    float2 UVLeftTop;
    float2 UVDimensions;
    uint MipLevel;
    float LayerOrDepth;
};

#if COMPILE_FullscreenVert
vertex VertexToFragment FullscreenVert(uint vI [[vertex_id]]) {
   float2 inTex = float2((vI << 1) & 2, vI & 2);
   VertexToFragment out;
   out.tex = inTex;
   out.pos = float4(inTex * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
   return out;
}
#endif

#if COMPILE_BlitFrom2D
fragment float4 BlitFrom2D(
    VertexToFragment input [[stage_in]],
    constant SourceRegion &sourceRegion [[buffer(0)]],
    texture2d<float> sourceTexture [[texture(0)]],
    sampler sourceSampler [[sampler(0)]])
{
    float2 newCoord = sourceRegion.UVLeftTop + sourceRegion.UVDimensions * input.tex;
    return sourceTexture.sample(sourceSampler, newCoord, level(sourceRegion.MipLevel));
}
#endif

#if COMPILE_BlitFrom2DArray
fragment float4 BlitFrom2DArray(
    VertexToFragment input [[stage_in]],
    constant SourceRegion &sourceRegion [[buffer(0)]],
    texture2d_array<float> sourceTexture [[texture(0)]],
    sampler sourceSampler [[sampler(0)]])
{
    float2 newCoord = sourceRegion.UVLeftTop + sourceRegion.UVDimensions * input.tex;
    return sourceTexture.sample(sourceSampler, newCoord, (uint)sourceRegion.LayerOrDepth, level(sourceRegion.MipLevel));
}
#endif

#if COMPILE_BlitFrom3D
fragment float4 BlitFrom3D(
    VertexToFragment input [[stage_in]],
    constant SourceRegion &sourceRegion [[buffer(0)]],
    texture3d<float> sourceTexture [[texture(0)]],
    sampler sourceSampler [[sampler(0)]])
{
    float2 newCoord = sourceRegion.UVLeftTop + sourceRegion.UVDimensions * input.tex;
    return sourceTexture.sample(sourceSampler, float3(newCoord, sourceRegion.LayerOrDepth), level(sourceRegion.MipLevel));
}
#endif

#if COMPILE_BlitFromCube
fragment float4 BlitFromCube(
    VertexToFragment input [[stage_in]],
    constant SourceRegion &sourceRegion [[buffer(0)]],
    texturecube<float> sourceTexture [[texture(0)]],
    sampler sourceSampler [[sampler(0)]])
{
    // Thanks, Wikipedia! https://en.wikipedia.org/wiki/Cube_mapping
    float2 scaledUV = sourceRegion.UVLeftTop + sourceRegion.UVDimensions * input.tex;
    float u = 2.0 * scaledUV.x - 1.0;
    float v = 2.0 * scaledUV.y - 1.0;
    float3 newCoord;
    switch ((uint)sourceRegion.LayerOrDepth) {
        case 0: newCoord = float3(1.0, -v, -u); break; // POSITIVE X
        case 1: newCoord = float3(-1.0, -v, u); break; // NEGATIVE X
        case 2: newCoord = float3(u, 1.0, -v); break; // POSITIVE Y
        case 3: newCoord = float3(u, -1.0, v); break; // NEGATIVE Y
        case 4: newCoord = float3(u, -v, 1.0); break; // POSITIVE Z
        case 5: newCoord = float3(-u, -v, -1.0); break; // NEGATIVE Z
        default: newCoord = float3(0, 0, 0); break; // silences warning
    }
    return sourceTexture.sample(sourceSampler, newCoord, level(sourceRegion.MipLevel));
}
#endif

#if COMPILE_BlitFromCubeArray
fragment float4 BlitFromCubeArray(
    VertexToFragment input [[stage_in]],
    constant SourceRegion &sourceRegion [[buffer(0)]],
    texturecube_array<float> sourceTexture [[texture(0)]],
    sampler sourceSampler [[sampler(0)]])
{
    // Thanks, Wikipedia! https://en.wikipedia.org/wiki/Cube_mapping
    float2 scaledUV = sourceRegion.UVLeftTop + sourceRegion.UVDimensions * input.tex;
    float u = 2.0 * scaledUV.x - 1.0;
    float v = 2.0 * scaledUV.y - 1.0;
    float3 newCoord;
    switch (((uint)sourceRegion.LayerOrDepth) % 6) {
        case 0: newCoord = float3(1.0, -v, -u); break; // POSITIVE X
        case 1: newCoord = float3(-1.0, -v, u); break; // NEGATIVE X
        case 2: newCoord = float3(u, 1.0, -v); break; // POSITIVE Y
        case 3: newCoord = float3(u, -1.0, v); break; // NEGATIVE Y
        case 4: newCoord = float3(u, -v, 1.0); break; // POSITIVE Z
        case 5: newCoord = float3(-u, -v, -1.0); break; // NEGATIVE Z
        default: newCoord = float3(0, 0, 0); break; // silences warning
    }
    return sourceTexture.sample(sourceSampler, newCoord, (uint)sourceRegion.LayerOrDepth / 6, level(sourceRegion.MipLevel));
}
#endif

void main(out float4 pos : SV_Position,
          out float2 clip[2] : SV_ClipDistance,  // array of vector float
          out float2 cull[2] : SV_CullDistance)  // array of vector float
{
    pos = 1.0f.xxxx;
    clip[0].x = 0.5f;
    clip[0].y = 0.6f;
    clip[1].x = 0.7f;
    clip[1].y = 0.8f;

    cull[0].x = 0.525f;
    cull[0].y = 0.625f;
    cull[1].x = 0.725f;
    cull[1].y = 0.825f;
}

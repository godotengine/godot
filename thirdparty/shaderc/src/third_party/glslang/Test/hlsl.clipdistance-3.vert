void main(out float4 pos : SV_Position,
          out float clip[2] : SV_ClipDistance, // array of scalar float
          out float cull[2] : SV_CullDistance) // array of scalar float
{
    pos = 1.0f.xxxx;
    clip[0] = 0.5f;
    clip[1] = 0.6f;

    cull[0] = 0.525f;
    cull[1] = 0.625f;
}



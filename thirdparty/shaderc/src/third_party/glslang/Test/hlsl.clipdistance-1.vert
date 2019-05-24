void main(out float4 pos : SV_Position, 
          out float clip : SV_ClipDistance,  // scalar float
          out float cull : SV_CullDistance)  // scalar float
{
    pos = 1.0f.xxxx;
    clip = 0.5f;
    cull = 0.51f;
}

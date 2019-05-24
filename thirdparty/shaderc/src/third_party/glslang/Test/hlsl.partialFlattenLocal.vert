Texture2D tex;

struct Packed {
  Texture2D     tex;
  float3        pos[3];
  float2        uv[2];
  float         x;
  int           n;
};

float4 main(float4 pos : POSITION) : SV_POSITION
{
  Packed packed;
  packed.tex    = tex;
  packed.pos[0] = float3(0, 0, 0);
  packed.uv[0]  = float2(0, 1);
  packed.x      = 1.0;
  packed.n      = 3;

  for (int i = 0; i < 1; ++i) {
    packed.pos[i].xy += packed.uv[i];
  }

  Packed packed2 = packed;

  return pos + float4(packed2.pos[0], 0);
}
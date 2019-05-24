SamplerState       g_sSamp : register(s0);

uniform Texture1D <float4> g_tTex1df4 : register(t0);

struct VS_OUTPUT
{
    float4 Pos : SV_Position;
};

VS_OUTPUT main()
{
   VS_OUTPUT vsout;

   uint WidthU;
   uint NumberOfLevelsU;

   // Most of the tests are in the hlsl.getdimensions.dx10.frag on the fragment side.
   // This is just to establish that GetDimensions appears in the vertex stage.

   // 1D, float tx, uint params
   g_tTex1df4 . GetDimensions(WidthU);
   g_tTex1df4 . GetDimensions(6, WidthU, NumberOfLevelsU);

   vsout.Pos = float4(0,0,0,0);

   return vsout;
}

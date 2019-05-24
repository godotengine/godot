SamplerState       g_sSamp : register(s0);

Texture1DArray          g_tTex1df4a : register(t1);

uniform Texture1DArray <float4> g_tTex1df4 : register(t0);
Texture1DArray <int4>   g_tTex1di4;
Texture1DArray <uint4>  g_tTex1du4;

Texture2DArray <float4> g_tTex2df4;
Texture2DArray <int4>   g_tTex2di4;
Texture2DArray <uint4>  g_tTex2du4;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   // No 1D gathers

   float4 txval20 = g_tTex2df4 . Gather(g_sSamp, float3(0.1, 0.2, 0.3), int2(1,0));
   int4   txval21 = g_tTex2di4 . Gather(g_sSamp, float3(0.3, 0.4, 0.4), int2(1,1));
   uint4  txval22 = g_tTex2du4 . Gather(g_sSamp, float3(0.5, 0.6, 0.7), int2(1,-1));

   // No 3D gathers
   // No Cube offset gathers

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

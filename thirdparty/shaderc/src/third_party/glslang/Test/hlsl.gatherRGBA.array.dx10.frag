SamplerState       g_sSamp : register(s0);
uniform sampler2D          g_sSamp2d;

uniform Texture1DArray <float4> g_tTex1df4a : register(t0);
Texture1DArray <int4>   g_tTex1di4a;
Texture1DArray <uint4>  g_tTex1du4a;

Texture2DArray <float4> g_tTex2df4a;
Texture2DArray <int4>   g_tTex2di4a;
Texture2DArray <uint4>  g_tTex2du4a;

TextureCubeArray <float4> g_tTexcdf4a;
TextureCubeArray <int4>   g_tTexcdi4a;
TextureCubeArray <uint4>  g_tTexcdu4a;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

uniform float  c1;
uniform float2 c2;
uniform float3 c3;
uniform float4 c4;

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   // no 1D gathers

   float4 txval00 = g_tTex2df4a . GatherRed(g_sSamp, c3);
   int4   txval01 = g_tTex2di4a . GatherRed(g_sSamp, c3);
   uint4  txval02 = g_tTex2du4a . GatherRed(g_sSamp, c3);

   float4 txval10 = g_tTex2df4a . GatherGreen(g_sSamp, c3);
   int4   txval11 = g_tTex2di4a . GatherGreen(g_sSamp, c3);
   uint4  txval12 = g_tTex2du4a . GatherGreen(g_sSamp, c3);

   float4 txval20 = g_tTex2df4a . GatherBlue(g_sSamp, c3);
   int4   txval21 = g_tTex2di4a . GatherBlue(g_sSamp, c3);
   uint4  txval22 = g_tTex2du4a . GatherBlue(g_sSamp, c3);

   float4 txval30 = g_tTex2df4a . GatherAlpha(g_sSamp, c3);
   int4   txval31 = g_tTex2di4a . GatherAlpha(g_sSamp, c3);
   uint4  txval32 = g_tTex2du4a . GatherAlpha(g_sSamp, c3);

   // no 3D gathers

   float4 txval40 = g_tTexcdf4a . GatherRed(g_sSamp, c4);
   int4   txval41 = g_tTexcdi4a . GatherRed(g_sSamp, c4);
   uint4  txval42 = g_tTexcdu4a . GatherRed(g_sSamp, c4);

   float4 txval50 = g_tTexcdf4a . GatherGreen(g_sSamp, c4);
   int4   txval51 = g_tTexcdi4a . GatherGreen(g_sSamp, c4);
   uint4  txval52 = g_tTexcdu4a . GatherGreen(g_sSamp, c4);

   float4 txval60 = g_tTexcdf4a . GatherBlue(g_sSamp, c4);
   int4   txval61 = g_tTexcdi4a . GatherBlue(g_sSamp, c4);
   uint4  txval62 = g_tTexcdu4a . GatherBlue(g_sSamp, c4);

   float4 txval70 = g_tTexcdf4a . GatherAlpha(g_sSamp, c4);
   int4   txval71 = g_tTexcdi4a . GatherAlpha(g_sSamp, c4);
   uint4  txval72 = g_tTexcdu4a . GatherAlpha(g_sSamp, c4);

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

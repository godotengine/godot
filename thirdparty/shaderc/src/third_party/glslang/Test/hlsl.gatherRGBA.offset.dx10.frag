SamplerState       g_sSamp : register(s0);
uniform sampler2D          g_sSamp2d;

Texture1D          g_tTex1df4a : register(t1);

uniform Texture1D <float4> g_tTex1df4 : register(t0);
Texture1D <int4>   g_tTex1di4;
Texture1D <uint4>  g_tTex1du4;

Texture2D <float4> g_tTex2df4;
Texture2D <int4>   g_tTex2di4;
Texture2D <uint4>  g_tTex2du4;

Texture3D <float4> g_tTex3df4;
Texture3D <int4>   g_tTex3di4;
Texture3D <uint4>  g_tTex3du4;

TextureCube <float4> g_tTexcdf4;
TextureCube <int4>   g_tTexcdi4;
TextureCube <uint4>  g_tTexcdu4;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

uniform float  c1;
uniform float2 c2;
uniform float3 c3;
uniform float4 c4;

uniform int  o1;
uniform int2 o2;
uniform int3 o3;
uniform int4 o4;

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   uint status;

   // no 1D gathers

   float4 txval001 = g_tTex2df4 . GatherRed(g_sSamp, c2, o2);
   int4   txval011 = g_tTex2di4 . GatherRed(g_sSamp, c2, o2);
   uint4  txval021 = g_tTex2du4 . GatherRed(g_sSamp, c2, o2);

   float4 txval004 = g_tTex2df4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2);
   int4   txval014 = g_tTex2di4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2);
   uint4  txval024 = g_tTex2du4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2);
   
   // float4 txval00s = g_tTex2df4 . GatherRed(g_sSamp, c2, o2, status);
   // int4   txval01s = g_tTex2di4 . GatherRed(g_sSamp, c2, o2, status);
   // uint4  txval02s = g_tTex2du4 . GatherRed(g_sSamp, c2, o2, status);

   // float4 txval004s = g_tTex2df4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2, status);
   // int4   txval014s = g_tTex2di4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2, status);
   // uint4  txval024s = g_tTex2du4 . GatherRed(g_sSamp, c2, o2, o2, o2, o2, status);

   float4 txval101 = g_tTex2df4 . GatherGreen(g_sSamp, c2, o2);
   int4   txval111 = g_tTex2di4 . GatherGreen(g_sSamp, c2, o2);
   uint4  txval121 = g_tTex2du4 . GatherGreen(g_sSamp, c2, o2);

   float4 txval104 = g_tTex2df4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2);
   int4   txval114 = g_tTex2di4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2);
   uint4  txval124 = g_tTex2du4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2);

   // float4 txval10s = g_tTex2df4 . GatherGreen(g_sSamp, c2, o2, status);
   // int4   txval11s = g_tTex2di4 . GatherGreen(g_sSamp, c2, o2, status);
   // uint4  txval12s = g_tTex2du4 . GatherGreen(g_sSamp, c2, o2, status);

   // float4 txval104 = g_tTex2df4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2, status);
   // int4   txval114 = g_tTex2di4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2, status);
   // uint4  txval124 = g_tTex2du4 . GatherGreen(g_sSamp, c2, o2, o2, o2, o2, status);

   float4 txval201 = g_tTex2df4 . GatherBlue(g_sSamp, c2, o2);
   int4   txval211 = g_tTex2di4 . GatherBlue(g_sSamp, c2, o2);
   uint4  txval221 = g_tTex2du4 . GatherBlue(g_sSamp, c2, o2);

   float4 txval204 = g_tTex2df4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2);
   int4   txval214 = g_tTex2di4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2);
   uint4  txval224 = g_tTex2du4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2);

   // float4 txval204s = g_tTex2df4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2, status);
   // int4   txval214s = g_tTex2di4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2, status);
   // uint4  txval224s = g_tTex2du4 . GatherBlue(g_sSamp, c2, o2, o2, o2, o2, status);

   // float4 txval20s = g_tTex2df4 . GatherBlue(g_sSamp, c2, o2, status);
   // int4   txval21s = g_tTex2di4 . GatherBlue(g_sSamp, c2, o2, status);
   // uint4  txval22s = g_tTex2du4 . GatherBlue(g_sSamp, c2, o2, status);

   float4 txval301 = g_tTex2df4 . GatherAlpha(g_sSamp, c2, o2);
   int4   txval311 = g_tTex2di4 . GatherAlpha(g_sSamp, c2, o2);
   uint4  txval321 = g_tTex2du4 . GatherAlpha(g_sSamp, c2, o2);

   float4 txval304 = g_tTex2df4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2);
   int4   txval314 = g_tTex2di4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2);
   uint4  txval324 = g_tTex2du4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2);

   // float4 txval304s = g_tTex2df4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2, status);
   // int4   txval314s = g_tTex2di4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2, status);
   // uint4  txval324s = g_tTex2du4 . GatherAlpha(g_sSamp, c2, o2, o2, o2, o2, status);

   // float4 txval30s = g_tTex2df4 . GatherAlpha(g_sSamp, c2, o2, status);
   // int4   txval31s = g_tTex2di4 . GatherAlpha(g_sSamp, c2, o2, status);
   // uint4  txval32s = g_tTex2du4 . GatherAlpha(g_sSamp, c2, o2, status);

   // no 3D gathers with offset

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

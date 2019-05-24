SamplerComparisonState  g_sSampCmp : register(s0);

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

   float4 txval80 = g_tTex2df4a . GatherCmp(g_sSampCmp, c3, .75);
   int4   txval81 = g_tTex2di4a . GatherCmp(g_sSampCmp, c3, .75);
   uint4  txval82 = g_tTex2du4a . GatherCmp(g_sSampCmp, c3, .75);

   float4 txval00 = g_tTex2df4a . GatherCmpRed(g_sSampCmp, c3, .75);
   int4   txval01 = g_tTex2di4a . GatherCmpRed(g_sSampCmp, c3, .75);
   uint4  txval02 = g_tTex2du4a . GatherCmpRed(g_sSampCmp, c3, .75);

   float4 txval10 = g_tTex2df4a . GatherCmpGreen(g_sSampCmp, c3, .75);
   int4   txval11 = g_tTex2di4a . GatherCmpGreen(g_sSampCmp, c3, .75);
   uint4  txval12 = g_tTex2du4a . GatherCmpGreen(g_sSampCmp, c3, .75);

   float4 txval20 = g_tTex2df4a . GatherCmpBlue(g_sSampCmp, c3, .75);
   int4   txval21 = g_tTex2di4a . GatherCmpBlue(g_sSampCmp, c3, .75);
   uint4  txval22 = g_tTex2du4a . GatherCmpBlue(g_sSampCmp, c3, .75);

   float4 txval30 = g_tTex2df4a . GatherCmpAlpha(g_sSampCmp, c3, .75);
   int4   txval31 = g_tTex2di4a . GatherCmpAlpha(g_sSampCmp, c3, .75);
   uint4  txval32 = g_tTex2du4a . GatherCmpAlpha(g_sSampCmp, c3, .75);

   // no 3D gathers

   float4 txval40 = g_tTexcdf4a . GatherCmpRed(g_sSampCmp, c4, .75);
   int4   txval41 = g_tTexcdi4a . GatherCmpRed(g_sSampCmp, c4, .75);
   uint4  txval42 = g_tTexcdu4a . GatherCmpRed(g_sSampCmp, c4, .75);

   float4 txval50 = g_tTexcdf4a . GatherCmpGreen(g_sSampCmp, c4, .75);
   int4   txval51 = g_tTexcdi4a . GatherCmpGreen(g_sSampCmp, c4, .75);
   uint4  txval52 = g_tTexcdu4a . GatherCmpGreen(g_sSampCmp, c4, .75);

   float4 txval60 = g_tTexcdf4a . GatherCmpBlue(g_sSampCmp, c4, .75);
   int4   txval61 = g_tTexcdi4a . GatherCmpBlue(g_sSampCmp, c4, .75);
   uint4  txval62 = g_tTexcdu4a . GatherCmpBlue(g_sSampCmp, c4, .75);

   float4 txval70 = g_tTexcdf4a . GatherCmpAlpha(g_sSampCmp, c4, .75);
   int4   txval71 = g_tTexcdi4a . GatherCmpAlpha(g_sSampCmp, c4, .75);
   uint4  txval72 = g_tTexcdu4a . GatherCmpAlpha(g_sSampCmp, c4, .75);

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

SamplerState       g_sSamp : register(s0);

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

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   float4 txval10 = g_tTex1df4 . SampleGrad(g_sSamp, 0.1, 1.1, 1.2);
   int4   txval11 = g_tTex1di4 . SampleGrad(g_sSamp, 0.2, 1.1, 1.2);
   uint4  txval12 = g_tTex1du4 . SampleGrad(g_sSamp, 0.3, 1.1, 1.2);

   float4 txval20 = g_tTex2df4 . SampleGrad(g_sSamp, float2(0.1, 0.2), float2(1.1, 1.2), float2(1.1, 1.2));
   int4   txval21 = g_tTex2di4 . SampleGrad(g_sSamp, float2(0.3, 0.4), float2(1.1, 1.2), float2(1.1, 1.2));
   uint4  txval22 = g_tTex2du4 . SampleGrad(g_sSamp, float2(0.5, 0.6), float2(1.1, 1.2), float2(1.1, 1.2));

   float4 txval30 = g_tTex3df4 . SampleGrad(g_sSamp, float3(0.1, 0.2, 0.3), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));
   int4   txval31 = g_tTex3di4 . SampleGrad(g_sSamp, float3(0.4, 0.5, 0.6), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));
   uint4  txval32 = g_tTex3du4 . SampleGrad(g_sSamp, float3(0.7, 0.8, 0.9), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));

   float4 txval40 = g_tTexcdf4 . SampleGrad(g_sSamp, float3(0.1, 0.2, 0.3), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));
   int4   txval41 = g_tTexcdi4 . SampleGrad(g_sSamp, float3(0.4, 0.5, 0.6), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));
   uint4  txval42 = g_tTexcdu4 . SampleGrad(g_sSamp, float3(0.7, 0.8, 0.9), float3(1.1, 1.2, 1.3), float3(1.1, 1.2, 1.3));

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

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

struct VS_OUTPUT
{
    float4 Pos : SV_Position;
};

VS_OUTPUT main()
{
   VS_OUTPUT vsout;

   // no 1D gathers

   float4 txval20 = g_tTex2df4 . Gather(g_sSamp, float2(0.1, 0.2));
   int4   txval21 = g_tTex2di4 . Gather(g_sSamp, float2(0.3, 0.4));
   uint4  txval22 = g_tTex2du4 . Gather(g_sSamp, float2(0.5, 0.6));

   // no 3D gathers

   float4 txval40 = g_tTexcdf4 . Gather(g_sSamp, float3(0.1, 0.2, 0.3));
   int4   txval41 = g_tTexcdi4 . Gather(g_sSamp, float3(0.4, 0.5, 0.6));
   uint4  txval42 = g_tTexcdu4 . Gather(g_sSamp, float3(0.7, 0.8, 0.9));

   vsout.Pos = float4(0,0,0,0);

   return vsout;
}

SamplerState       g_sSamp : register(s0);

RWTexture1D <float4> g_tTex1df4 : register(t0);
RWTexture1D <int4>   g_tTex1di4;
RWTexture1D <uint4>  g_tTex1du4;

RWTexture2D <float4> g_tTex2df4;
RWTexture2D <int4>   g_tTex2di4;
RWTexture2D <uint4>  g_tTex2du4;

RWTexture3D <float4> g_tTex3df4;
RWTexture3D <int4>   g_tTex3di4;
RWTexture3D <uint4>  g_tTex3du4;

RWTexture1DArray <float4> g_tTex1df4a;
RWTexture1DArray <int4>   g_tTex1di4a;
RWTexture1DArray <uint4>  g_tTex1du4a;

RWTexture2DArray <float4> g_tTex2df4a;
RWTexture2DArray <int4>   g_tTex2di4a;
RWTexture2DArray <uint4>  g_tTex2du4a;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

uniform int   c1;
uniform int2  c2;
uniform int3  c3;
uniform int4  c4;

uniform int   o1;
uniform int2  o2;
uniform int3  o3;
uniform int4  o4;

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   // 1D
   g_tTex1df4a.Load(c2);
   g_tTex1di4a.Load(c2);
   g_tTex1du4a.Load(c2);

   // 2D
   g_tTex2df4a.Load(c3);
   g_tTex2di4a.Load(c3);
   g_tTex2du4a.Load(c3);

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

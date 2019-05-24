SamplerState       g_sSamp : register(s0);

Texture2DMS <float4> g_tTex2dmsf4;
Texture2DMS <int4>   g_tTex2dmsi4;
Texture2DMS <uint4>  g_tTex2dmsu4;

Texture2DMSArray <float4> g_tTex2dmsf4a;
Texture2DMSArray <int4>   g_tTex2dmsi4a;
Texture2DMSArray <uint4>  g_tTex2dmsu4a;

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

   // 2DMS, no offset
   g_tTex2dmsf4.Load(c2, 3);
   g_tTex2dmsi4.Load(c2, 3);
   g_tTex2dmsu4.Load(c2, 3);

   // 2DMS, offset
   g_tTex2dmsf4.Load(c2, 3, o2);
   g_tTex2dmsi4.Load(c2, 3, o2);
   g_tTex2dmsu4.Load(c2, 3, o2);

   // 2DMSArray, no offset
   g_tTex2dmsf4a.Load(c3, 3);
   g_tTex2dmsi4a.Load(c3, 3);
   g_tTex2dmsu4a.Load(c3, 3);

   // 2DMSArray, offset
   g_tTex2dmsf4a.Load(c3, 3, o2);
   g_tTex2dmsi4a.Load(c3, 3, o2);
   g_tTex2dmsu4a.Load(c3, 3, o2);

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}


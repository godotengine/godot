SamplerState       g_sSamp : register(s0);

Texture1D <float4> g_tTex1df4 : register(t0);
Texture1D <int4>   g_tTex1di4;
Texture1D <uint4>  g_tTex1du4;

Texture2D <float4> g_tTex2df4;
Texture2D <int4>   g_tTex2di4;
Texture2D <uint4>  g_tTex2du4;

Texture3D <float4> g_tTex3df4;
Texture3D <int4>   g_tTex3di4;
Texture3D <uint4>  g_tTex3du4;

Texture1DArray <float4> g_tTex1df4a;
Texture1DArray <int4>   g_tTex1di4a;
Texture1DArray <uint4>  g_tTex1du4a;

Texture2DArray <float4> g_tTex2df4a;
Texture2DArray <int4>   g_tTex2di4a;
Texture2DArray <uint4>  g_tTex2du4a;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform int   c1;
uniform int2  c2;
uniform int3  c3;
uniform int4  c4;

uniform int   o1;
uniform int2  o2;
uniform int3  o3;
uniform int4  o4;

int4   Fn1(in int4 x)   { return x; }
uint4  Fn1(in uint4 x)  { return x; }
float4 Fn1(in float4 x) { return x; }

float4 SomeValue() { return c4; }

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   // 1D
   g_tTex1df4[c1];
   
   float4 r00 = g_tTex1df4[c1];
   int4   r01 = g_tTex1di4[c1];
   uint4  r02 = g_tTex1du4[c1];

   // 2D
   float4 r10 = g_tTex2df4[c2];
   int4   r11 = g_tTex2di4[c2];
   uint4  r12 = g_tTex2du4[c2];
   
   // 3D
   float4 r20 = g_tTex3df4[c3];
   int4   r21 = g_tTex3di4[c3];
   uint4  r22 = g_tTex3du4[c3];

   // Test function calling
   Fn1(g_tTex1df4[c1]);  // in
   Fn1(g_tTex1di4[c1]);  // in
   Fn1(g_tTex1du4[c1]);  // in

   psout.Color = 1.0;

   return psout;
}

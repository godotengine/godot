SamplerState       g_sSamp : register(s0);

RWTexture1D <float2> g_tTex1df2;
RWTexture1D <int2>   g_tTex1di2;
RWTexture1D <uint2>  g_tTex1du2;

RWTexture2D <float2> g_tTex2df2;
RWTexture2D <int2>   g_tTex2di2;
RWTexture2D <uint2>  g_tTex2du2;

RWTexture3D <float2> g_tTex3df2;
RWTexture3D <int2>   g_tTex3di2;
RWTexture3D <uint2>  g_tTex3du2;

RWTexture1DArray <float2> g_tTex1df2a;
RWTexture1DArray <int2>   g_tTex1di2a;
RWTexture1DArray <uint2>  g_tTex1du2a;

RWTexture2DArray <float2> g_tTex2df2a;
RWTexture2DArray <int2>   g_tTex2di2a;
RWTexture2DArray <uint2>  g_tTex2du2a;

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

uniform float2 uf2;
uniform int2   ui2;
uniform uint2  uu2;

int2   Fn1(in int2 x)   { return x; }
uint2  Fn1(in uint2 x)  { return x; }
float2 Fn1(in float2 x) { return x; }

void Fn2(out int2 x)   { x = int2(0,0); }
void Fn2(out uint2 x)  { x = uint2(0,0); }
void Fn2(out float2 x) { x = float2(0,0); }

float2 SomeValue() { return c2; }

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   // 1D
   g_tTex1df2[c1];
   
   float2 r00 = g_tTex1df2[c1];
   int2   r01 = g_tTex1di2[c1];
   uint2  r02 = g_tTex1du2[c1];

   // 2D
   float2 r10 = g_tTex2df2[c2];
   int2   r11 = g_tTex2di2[c2];
   uint2  r12 = g_tTex2du2[c2];
   
   // 3D
   float2 r20 = g_tTex3df2[c3];
   int2   r21 = g_tTex3di2[c3];
   uint2  r22 = g_tTex3du2[c3];

   float2 lf2 = uf2;

   // Test as L-values
   // 1D
   g_tTex1df2[c1] = SomeValue(); // complex R-value
   g_tTex1df2[c1] = lf2;
   g_tTex1di2[c1] = int2(2,2);
   g_tTex1du2[c1] = uint2(3,2);

   // Test some operator= things, which need to do both a load and a store.
   float2 val1 = (g_tTex1df2[c1] *= 2.0);
   g_tTex1df2[c1] -= 3.0;
   g_tTex1df2[c1] += 4.0;
   
   g_tTex1di2[c1] /= 2;
   g_tTex1di2[c1] %= 2;
   g_tTex1di2[c1] &= 0xffff;
   g_tTex1di2[c1] |= 0xf0f0;
   g_tTex1di2[c1] <<= 2;
   g_tTex1di2[c1] >>= 2;

   // 2D
   g_tTex2df2[c2] = SomeValue(); // complex L-value
   g_tTex2df2[c2] = lf2;
   g_tTex2di2[c2] = int2(5,2);
   g_tTex2du2[c2] = uint2(6,2);
   
   // 3D
   g_tTex3df2[c3] = SomeValue(); // complex L-value
   g_tTex3df2[c3] = lf2;
   g_tTex3di2[c3] = int2(8,6);
   g_tTex3du2[c3] = uint2(9,2);

   // Test function calling
   Fn1(g_tTex1df2[c1]);  // in
   Fn1(g_tTex1di2[c1]);  // in
   Fn1(g_tTex1du2[c1]);  // in

   Fn2(g_tTex1df2[c1]);  // out
   Fn2(g_tTex1di2[c1]);  // out
   Fn2(g_tTex1du2[c1]);  // out

   // Test increment operators
   // pre-ops
   ++g_tTex1df2[c1];
   ++g_tTex1di2[c1];
   ++g_tTex1du2[c1];

   --g_tTex1df2[c1];
   --g_tTex1di2[c1];
   --g_tTex1du2[c1];

   // post-ops
   g_tTex1df2[c1]++;
   g_tTex1du2[c1]--;
   g_tTex1di2[c1]++;

   g_tTex1df2[c1]--;
   g_tTex1di2[c1]++;
   g_tTex1du2[c1]--;

   // read and write
   g_tTex1df2[1] = g_tTex2df2[int2(2,3)];

   psout.Color = 1.0;

   return psout;
}

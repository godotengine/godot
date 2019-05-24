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
};

uniform int   c1;
uniform int2  c2;
uniform int3  c3;
uniform int4  c4;

uniform int   o1;
uniform int2  o2;
uniform int3  o3;
uniform int4  o4;

uniform float4 uf4;
uniform int4   ui4;
uniform uint4  uu4;

int4   Fn1(in int4 x)   { return x; }
uint4  Fn1(in uint4 x)  { return x; }
float4 Fn1(in float4 x) { return x; }

void Fn2(out int4 x)   { x = int4(0); }
void Fn2(out uint4 x)  { x = uint4(0); }
void Fn2(out float4 x) { x = float4(0); }

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

   float4 lf4 = uf4;

   // Test as L-values
   // 1D
   g_tTex1df4[c1] = SomeValue(); // complex R-value
   g_tTex1df4[c1] = lf4;
   g_tTex1di4[c1] = int4(2,2,3,4);
   g_tTex1du4[c1] = uint4(3,2,3,4);

   // Test some operator= things, which need to do both a load and a store.
   float4 val1 = (g_tTex1df4[c1] *= 2.0);
   g_tTex1df4[c1] -= 3.0;
   g_tTex1df4[c1] += 4.0;
   
   g_tTex1di4[c1] /= 2;
   g_tTex1di4[c1] %= 2;
   g_tTex1di4[c1] &= 0xffff;
   g_tTex1di4[c1] |= 0xf0f0;
   g_tTex1di4[c1] <<= 2;
   g_tTex1di4[c1] >>= 2;

   // 2D
   g_tTex2df4[c2] = SomeValue(); // complex L-value
   g_tTex2df4[c2] = lf4;
   g_tTex2di4[c2] = int4(5,2,3,4);
   g_tTex2du4[c2] = uint4(6,2,3,4);
   
   // 3D
   g_tTex3df4[c3] = SomeValue(); // complex L-value
   g_tTex3df4[c3] = lf4;
   g_tTex3di4[c3] = int4(8,6,7,8);
   g_tTex3du4[c3] = uint4(9,2,3,4);

   // Test function calling
   Fn1(g_tTex1df4[c1]);  // in
   Fn1(g_tTex1di4[c1]);  // in
   Fn1(g_tTex1du4[c1]);  // in

   Fn2(g_tTex1df4[c1]);  // out
   Fn2(g_tTex1di4[c1]);  // out
   Fn2(g_tTex1du4[c1]);  // out

   // Test increment operators
   // pre-ops
   ++g_tTex1df4[c1];
   ++g_tTex1di4[c1];
   ++g_tTex1du4[c1];

   --g_tTex1df4[c1];
   --g_tTex1di4[c1];
   --g_tTex1du4[c1];

   // post-ops
   g_tTex1df4[c1]++;
   g_tTex1du4[c1]--;
   g_tTex1di4[c1]++;

   g_tTex1df4[c1]--;
   g_tTex1di4[c1]++;
   g_tTex1du4[c1]--;

   // read and write
   g_tTex1df4[1] = g_tTex2df4[int2(2,3)];

   psout.Color = 1.0;

   return psout;
}

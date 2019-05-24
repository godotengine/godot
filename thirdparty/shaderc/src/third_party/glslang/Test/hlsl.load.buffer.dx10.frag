uniform Buffer <float4> g_tTexbf4_test : register(t0);

Buffer          g_tTexbf4;  // default is float4
Buffer <int4>   g_tTexbi4;
Buffer <uint4>  g_tTexbu4;

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

   // Buffer
   float4 r00 = g_tTexbf4.Load(c1);
   int4   r01 = g_tTexbi4.Load(c1);
   uint4  r02 = g_tTexbu4.Load(c1);

   // TODO: other types that can be put in sampler buffers, like float2x2, and float3.

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

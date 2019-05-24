uniform Buffer <float> g_tTexbfs_test : register(t0);

Buffer <float> g_tTexbfs;
Buffer <int>   g_tTexbis;
Buffer <uint>  g_tTexbus;

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
   float r00 = g_tTexbfs.Load(c1);
   int   r01 = g_tTexbis.Load(c1);
   uint  r02 = g_tTexbus.Load(c1);

   // TODO: other types that can be put in sampler buffers, like float2x2, and float3.

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

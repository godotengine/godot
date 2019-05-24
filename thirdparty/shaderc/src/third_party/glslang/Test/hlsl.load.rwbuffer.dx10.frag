
RWBuffer <float4> g_tBuffF;
RWBuffer <int4>   g_tBuffI;
RWBuffer <uint4>  g_tBuffU;

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

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   g_tBuffF.Load(c1);
   g_tBuffU.Load(c1);
   g_tBuffI.Load(c1);

   psout.Color = 1.0;

   return psout;
}

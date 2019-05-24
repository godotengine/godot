SamplerState       g_sSamp : register(s0);

Texture2DMS      <float4>  g_tTex2dmsf4;
Texture2DMSArray <float4>  g_tTex2dmsf4a;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

PS_OUTPUT main(int sample : SAMPLE)
{
   PS_OUTPUT psout;

   float2 r00 = g_tTex2dmsf4.GetSamplePosition(sample);
   float2 r01 = g_tTex2dmsf4a.GetSamplePosition(sample);

   psout.Color = 1.0;
   psout.Depth = 1.0;

   return psout;
}

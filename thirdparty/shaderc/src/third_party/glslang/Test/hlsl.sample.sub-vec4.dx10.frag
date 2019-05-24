SamplerState       g_sSamp : register(s0);

Texture1D <float>  g_tTex1df1;
Texture1D <float2> g_tTex1df2;
Texture1D <float3> g_tTex1df3;
Texture1D <float4> g_tTex1df4;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
   PS_OUTPUT psout;

   float  txval10 = g_tTex1df1 . Sample(g_sSamp, 0.1);
   float2 txval11 = g_tTex1df2 . Sample(g_sSamp, 0.2);
   float3 txval12 = g_tTex1df3 . Sample(g_sSamp, 0.2);
   float4 txval13 = g_tTex1df4 . Sample(g_sSamp, 0.2);

   psout.Color = 1.0;
   return psout;
}


RWTexture1D <float> g_tTex1df1 : register(u2);
RWBuffer <uint>     g_tBuf1du1 : register(U3);

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
   float r00 = g_tTex1df1[0];
   uint  r01 = g_tBuf1du1[0];

   PS_OUTPUT psout;
   psout.Color = 1.0;
   return psout;
}

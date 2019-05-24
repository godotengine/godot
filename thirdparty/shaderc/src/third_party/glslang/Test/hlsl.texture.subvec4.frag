
Texture2DMS <float>  g_tTex2dmsf1;
Texture2DMS <float2> g_tTex2dmsf2;
Texture2DMS <float3> g_tTex2dmsf3;
Texture2DMS <float4> g_tTex2dmsf4;

Texture2D <float>  g_tTex2df1;
Texture2D <float2> g_tTex2df2;
Texture2D <float3> g_tTex2df3;
Texture2D <float4> g_tTex2df4;

SamplerState g_sSamp;

float4 main() : SV_Target0
{
    uint MipLevel;
    uint WidthU;
    uint HeightU;
    uint ElementsU;
    uint DepthU;
    uint NumberOfLevelsU;
    uint NumberOfSamplesU;

    g_tTex2dmsf1 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);
    g_tTex2dmsf2 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);
    g_tTex2dmsf3 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);
    g_tTex2dmsf4 . GetDimensions(WidthU, HeightU, NumberOfSamplesU);

    g_tTex2dmsf1 . Load(int2(1,2), 3);
    g_tTex2dmsf2 . Load(int2(1,2), 3);
    g_tTex2dmsf3 . Load(int2(1,2), 3);
    g_tTex2dmsf4 . Load(int2(1,2), 3);

    g_tTex2df1 . Sample(g_sSamp, float2(.1, .2));
    g_tTex2df2 . Sample(g_sSamp, float2(.1, .2));
    g_tTex2df3 . Sample(g_sSamp, float2(.1, .2));
    g_tTex2df4 . Sample(g_sSamp, float2(.1, .2));
    
    return 0;
}


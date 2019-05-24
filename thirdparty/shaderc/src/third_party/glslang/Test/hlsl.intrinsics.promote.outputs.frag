
struct PS_OUTPUT { float4 color : SV_Target0; };

int    i;
uint   u;
float  f;
bool   b;

int2   i2;
uint2  u2;
float2 f2;
bool2  b2;

Buffer    <float>  g_tTexbfs;
Texture1D <float4> g_tTex1df4;
uint  upos;
float fpos;

PS_OUTPUT main()
{
    int MipLevel;

    uint WidthU;
    uint HeightU;
    uint ElementsU;
    uint DepthU;
    uint NumberOfLevelsU;
    uint NumberOfSamplesU;

    int  WidthI;
    int  HeightI;
    int  ElementsI;
    int  DepthI;
    int  NumberOfLevelsI;
    int  NumberOfSamplesI;

    saturate(fpos);

    // Test output promotions
    g_tTex1df4 . GetDimensions(WidthI);
    g_tTex1df4 . GetDimensions(6, WidthI, NumberOfLevelsU);
    g_tTex1df4 . GetDimensions(6, WidthU, NumberOfLevelsI);
    g_tTex1df4 . GetDimensions(6, WidthI, NumberOfLevelsI);

    // max(i2, f2);
    PS_OUTPUT ps_output;
    ps_output.color = 0;
    return ps_output;
};

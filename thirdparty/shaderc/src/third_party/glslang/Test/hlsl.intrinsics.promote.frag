
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
    // Same shapes:

    float r00 = max(b,  f);
    uint  r01 = max(b,  u);
    int   r02 = max(b,  i);
    float r03 = max(i,  f);
    float r04 = max(u,  f);

    float2 r10 = max(b2,  f2);
    uint2  r11 = max(b2,  u2);
    int2   r12 = max(b2,  i2);
    float2 r13 = max(i2,  f2);
    float2 r14 = max(u2,  f2);

    float2 r20 = clamp(i2, u2, f2);  // 3 args, converts all to best type.
    uint2  r21 = clamp(b2, u2, b2);
    float2 r22 = clamp(b2, f2, b2);

    // Mixed shapes:
    float2 r30 = max(b,  f2);
    uint2  r31 = max(b,  u2);
    int2   r32 = max(b,  i2);
    float2 r33 = max(i,  f2);
    float2 r34 = max(u,  f2);

    float2 r40 = clamp(i, u2, f2);  // 3 args, converts all to best type.
    uint2  r41 = clamp(b2, u, b2);
    float2 r42 = clamp(b2, f, b);
    int2   r43 = clamp(i, i2, u2);

    float r50 = g_tTexbfs.Load(upos);
    float r51 = g_tTexbfs.Load(fpos);

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

    g_tTex1df4 . GetDimensions(WidthI);
    g_tTex1df4 . GetDimensions(6, WidthI, NumberOfLevelsU);
    g_tTex1df4 . GetDimensions(6, WidthU, NumberOfLevelsI);
    g_tTex1df4 . GetDimensions(6, WidthI, NumberOfLevelsI);

    // max(i2, f2);
    PS_OUTPUT ps_output;
    ps_output.color = r00;
    return ps_output;
};

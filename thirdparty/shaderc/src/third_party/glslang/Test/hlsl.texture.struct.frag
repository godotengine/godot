struct s1_t {
    float  c0;
    float2 c1;
    float  c2;
};

struct s2_t {
    float  c0;
    float3 c1;
};

struct s3_t {
    float2  c0;
    float1  c1;
};

struct s4_t {
    int  c0;
    int2 c1;
    int  c2;
};

struct s5_t {
    uint c0;
    uint c1;
};

SamplerState g_sSamp;
Texture2D <s1_t>   g_tTex2s1;
Texture2D <s2_t>   g_tTex2s2;
Texture2D <s3_t>   g_tTex2s3;
Texture2D <s4_t>   g_tTex2s4;
Texture2D <s5_t>   g_tTex2s5;

Texture2D <s1_t>   g_tTex2s1a; // same type as g_tTex2s1, to test fn signature matching.

// function overloading to test name mangling with textures templatized on structs
s1_t fn1(Texture2D <s1_t> t1) { return t1 . Sample(g_sSamp, float2(0.6, 0.61)); }
s2_t fn1(Texture2D <s2_t> t2) { return t2 . Sample(g_sSamp, float2(0.6, 0.61)); }

float4 main() : SV_Target0
{
    s1_t s1 = g_tTex2s1 . Sample(g_sSamp, float2(0.1, 0.11));
    s2_t s2 = g_tTex2s2 . Sample(g_sSamp, float2(0.2, 0.21));
    s3_t s3 = g_tTex2s3 . Sample(g_sSamp, float2(0.3, 0.31));
    s4_t s4 = g_tTex2s4 . Sample(g_sSamp, float2(0.4, 0.41));
    s5_t s5 = g_tTex2s5 . Sample(g_sSamp, float2(0.5, 0.51));

    s1_t r0 = fn1(g_tTex2s1);
    s2_t r1 = fn1(g_tTex2s2);
    s1_t r2 = fn1(g_tTex2s1a);

    return 0;
}


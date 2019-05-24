struct outs {
    int a;
    float b;
    bool c;
    float4 v;
};

static float4 gv = {0,0,1};
static float gfa[3] = {0,0};

struct Nest {
    float4x3 m;
    outs os;
    bool b;
};

outs PixelShaderFunction(float4 input) : COLOR0
{
    outs o2 = { 3 };
    outs o4;
    o4.v = gv * gfa[2];
    outs o1 = { };
    outs o3 = (outs)0;
    o4 = (outs)0;
    o4.c = o1.c;
    Nest nest = (Nest)0;

    float2 gf2a[4] = { };
    int cgi = { };
    o4.b = gf2a[2].y * cgi;

    return o4;
}

static const float2 cgf2a[3];
static const int ci;

struct {
};

struct {
    bool b;
};

struct myS {
    bool b, c;
    float4 a, d;
};

myS s1;

static class {
    float4 i;
} s2;

struct IN_S {
    linear float4 a;
    nointerpolation bool b;
    noperspective centroid float1 c;
    sample centroid float2 d;
    bool ff1 : SV_IsFrontFace;
    bool ff2 : packoffset(c0.y);
    bool ff3 : packoffset(c0.y) : register(ps_5_0, s0) ;
    float4 ff4 : VPOS : packoffset(c0.y) : register(ps_5_0, s0) <int bambam=30;> ;
};

float ff5 : packoffset(c101.y) : register(ps_5_0, s[5]);
float ff6 : packoffset(c102.y) : register(s3[5]);

struct empty {};

struct containEmpty {
    empty e;
};

float4 PixelShaderFunction(float4 input, IN_S s) : COLOR0
{
    class FS {
        bool3 b3;
    } s3;

    s3 == s3;
    s2.i = s.ff4;

    containEmpty ce;
    empty e;
    e = ce.e;

    return input;
}

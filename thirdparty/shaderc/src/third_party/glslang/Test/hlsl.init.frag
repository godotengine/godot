static float4 a1 = float4(1, 0.5, 0, 1), b1 = float4(2.0, 2.5, 2.1, 2.2);
static float4 a1i = {1, 0.5, 0, 1}, b1i = {2.0, 2.5, 2.1, 2.2};
static float a2 = 0.2, b2;
static float a3, b3 = 0.3;
static float a4, b4 = 0.4, c4;
static float a5 = 0.5, b5, c5 = 1.5;

struct Single1 { int f; };
static Single1 single1 = { 10 };

struct Single2 { uint2 v; };
static Single2 single2 = { { 1, 2 } };

struct Single3 { Single1 s1; };
static Single3 single3 = { { 3 } };

struct Single4 { Single2 s1; };
static Single4 single4 = { { { 4u, 5u } } };

float4 ShaderFunction(float4 input) : COLOR0
{
    float4 a2 = float4(0.2, 0.3, 0.4, 0.5);
    struct S1 {
        float f;
        int i;
    };
    struct S2 {
        int  j;
        float g;
        S1 s1;
    };
    S2 s2i = { 9, a5, { (a3,a4), 12} }, s2 = S2(9, a5, S1((a3,a4), 12));
    float a8 = (a2, b2), a9 = a5;

    return input * a1;
}

cbuffer Constants
{
    float a = 1.0f, b, c = 2.0f;
};

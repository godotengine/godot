// function selection under type conversion
void foo1(double a, bool b)  {}
void foo1(double a, uint b)  {}
void foo1(double a, int b)   {}
void foo1(double a, float b) {}
void foo1(double a, double b){}

// uint -> int
void foo2(int a, bool b)  {}
void foo2(int a, uint b)  {}
void foo2(int a, int b)   {}
void foo2(int a, float b) {}
void foo2(int a, double b){}

// everything can promote
void foo3(bool b)  {}
void foo4(uint b)  {}
void foo5(int b)   {}
void foo6(float b) {}
void foo7(double b){}

// shorter forward chain better than longer or backward chain
void foo8(float)  {}
void foo8(double) {}
void foo9(int)    {}
void foo9(uint)   {}
void foo10(bool)  {}
void foo10(int)   {}

// shape change is worse
void foo11(float3)  {}
void foo11(double)  {}
void foo11(int3)    {}
void foo11(uint)    {}
void foo12(float1)  {}
void foo12(double3) {}
void foo16(uint)    {}
void foo16(uint2)   {}

// shape change
void foo13(float3)  {}
void foo14(int1)     {}
void foo15(bool1)   {}

float4 PixelShaderFunction(float4 input) : COLOR0
{
    bool b;
    double d;
    uint u;
    int i;
    float f;

    foo1(d, b);
    foo1(d, d);
    foo1(d, u);
    foo1(d, i);
    foo1(d, f);

    foo1(f, b);
    foo1(f, d);
    foo1(f, u);
    foo1(f, i);
    foo1(f, f);

    foo1(u, b);
    foo1(u, d);
    foo1(u, u);
    foo1(u, i);
    foo1(u, f);

    foo1(i, b);
    foo1(i, d);
    foo1(i, u);
    foo1(i, i);
    foo1(i, f);

    foo2(u, b);
    foo2(u, d);
    foo2(u, u);
    foo2(u, i);
    foo2(u, f);

    foo2(i, b);
    foo2(i, d);
    foo2(i, u);
    foo2(i, i);
    foo2(i, f);

    foo3(b);
    foo3(d);
    foo3(u);
    foo3(i);
    foo3(f);

    foo4(b);
    foo4(d);
    foo4(u);
    foo4(i);
    foo4(f);

    foo5(b);
    foo5(d);
    foo5(u);
    foo5(i);
    foo5(f);

    foo6(b);
    foo6(d);
    foo6(u);
    foo6(i);
    foo6(f);

    foo7(b);
    foo7(d);
    foo7(u);
    foo7(i);
    foo7(f);

    foo8(b);
    foo8(u);
    foo8(i);

    foo9(b);
    foo9(f);
    foo9(d);

    foo10(u);
    foo10(f);
    foo10(d);

    foo11(b);
    foo11(f);
    foo12(float3(f));
    foo16(int2(i,i));

    foo13(f);
    foo14(int4(i));
    foo15(b);
    foo15(bool3(b));

    return input;
}

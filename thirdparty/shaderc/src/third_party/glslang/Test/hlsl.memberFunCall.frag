float method3(float a) { return 1.0; }

struct myContext {
    float method1() { return method2(); }
    float method2() { return method3(1.0); }
    float method3(float a) { return method4(a, a); }
    float method4(float a, float b) { return a + b + f; }
    float f;
};

float4 main() : SV_TARGET0
{
    myContext context;
    context.f = 3.0;
    return (float4)context.method1();
}

static float x = 1.0;

float f1()
{
    static float x = 2.0;
    x += 10.0;
    return x;
}

float f2(float p)
{
    static float x = 7.0;
    x += p;
    return x;
}

float4 main() : SV_TARGET
{
    return x + f1() + f1() + f2(5.0) + f2(x);
}

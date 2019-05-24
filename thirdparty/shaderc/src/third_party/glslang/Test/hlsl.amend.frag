float4 a;
float b;
static float4 m = a * b;
void f1()
{
    a * b;
}

float3 c;

void f2()
{
    a.x + b + c.x;
}

void f3()
{
    c;
}

int d;

void f4()
{
    d * a;
}

int e;
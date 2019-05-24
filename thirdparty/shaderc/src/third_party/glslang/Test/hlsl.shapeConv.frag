float4 PixelShaderFunction(float4 input, float f) : COLOR0
{
    float4 v;
    v = 1;
    v = 2.0;
    v = f;
    float3 u;
    u = float(1);
    u = float(2.0);
    u = float(f);
    float2 w = 2.0;
    float V = 1;
    float3 MyVal = V;

    float3 foo;
    foo > 4.0;
    foo >= 5.0;
    6.0 < foo;
    7.0 <= foo;

    all(v.x == v);
    any(f != v);

    float1 f1;

    f1 == v;
    v < f1;
    f1.x;
    f1.xxx;

    const float4 f4 = 3.0;

    uint ui;
    uint3 ui3;

    ui >> ui3;
    ui3 >> ui;

    v *= f1;
    f1 *= v;

    float3 mixed = u * v;
    f = u;
    f1 = u;
    float sf = v;
    float1 sf1 = v;

    return input * f4;
}

float4 c4;
float4 t4;
float4 f4;
float t;
float f;

float4 vectorCond()
{
    return (c4 ? t4 : f4) +
           (c4 ? t  : f ) +
           (t4 < f4 ? t4 : f4) +
           (c4 ? t : f4);
}

float4 scalarCond()
{
    float4 ret = t != f ? t * f4 : 1;
    return ret;
}

float2 fbSelect(bool2 cnd, float2 src0, float2 src1)
{
    return cnd ? src0 : src1;
}

float4 PixelShaderFunction(float4 input) : COLOR0
{
    int a = 1 < 2 ? 3 < 4 ? 5 : 6 : 7;
    int b = 1 < 2 ? 3 > 4 ? 5 : 6 : 7;
    int c = 1 > 2 ? 3 > 4 ? 5 : 6 : 7;
    int d = 1 > 2 ? 3 < 4 ? 5 : 6 : 7;
    float4 ret = a * input + 
                 b * input +
                 c * input +
                 d * input;
    int e;
    e = a = b ? c = d : 10, b = a ? d = c : 11;
    float4 f;
    f = ret.x < input.y ? c * input : d * input;
    return e * ret + f + vectorCond() + scalarCond() +
           float4(fbSelect(bool2(true, false), float2(1.0, 2.0), float2(3.0, 4.0)), 10.0, 10.0);
}

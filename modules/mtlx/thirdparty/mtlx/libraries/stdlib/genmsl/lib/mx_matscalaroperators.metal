float3x3 operator+(float3x3 a, float b)
{
    return a + float3x3(b,b,b,b,b,b,b,b,b);
}

float4x4 operator+(float4x4 a, float b)
{
    return a + float4x4(b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b);
}

float3x3 operator-(float3x3 a, float b)
{
    return a - float3x3(b,b,b,b,b,b,b,b,b);
}

float4x4 operator-(float4x4 a, float b)
{
    return a - float4x4(b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b);
}

float3x3 operator/(float3x3 a, float3x3 b)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            a[i][j] /= b[i][j];

    return a;
}

float4x4 operator/(float4x4 a, float4x4 b)
{
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            a[i][j] /= b[i][j];

    return a;
}

float3x3 operator/(float3x3 a, float b)
{
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            a[i][j] /= b;

    return a;
}

float4x4 operator/(float4x4 a, float b)
{
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            a[i][j] /= b;

    return a;
}

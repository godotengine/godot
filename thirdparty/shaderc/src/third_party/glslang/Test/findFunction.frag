#version 450

#extension GL_EXT_shader_explicit_arithmetic_types: enable

int64_t func(int8_t a, int16_t b, int16_t c)
{
    return int64_t(a | b + c);
}

int64_t func(int8_t a, int16_t b, int32_t c)
{
    return int64_t(a | b - c);
}

int64_t func(int32_t a, int32_t b, int32_t c)
{
    return int64_t(a / b + c);
}

int64_t func(float16_t a, float16_t b, float32_t c)
{
    return int64_t(a - b * c);
}

int64_t func(float16_t a, int16_t b, float32_t c)
{
    return int64_t(a - b * c);
}

void main()
{
    int8_t  x;
    int16_t y;
    int32_t z;
    int64_t w;
    float16_t f16;
    float64_t f64;
    int64_t b1 = func(x, y, z);
    int64_t b2 = func(y, y, z); // tie
    int64_t b3 = func(y, y, w); // No match
    int64_t b4 = func(y, z, f16); // No match
    int64_t b5 = func(y, y, f16);
    int64_t b7 = func(f16, f16, y);
    int64_t b8 = func(f16, f16, f64); // No match
    int64_t b9 = func(f16, x, f16); // tie
}

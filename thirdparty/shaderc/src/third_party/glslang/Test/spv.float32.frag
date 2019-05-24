#version 450

#extension GL_EXT_shader_explicit_arithmetic_types: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

void main()
{
}

// Single float literals
void literal()
{
    const float32_t f32c  = 0.000001f;
    const f32vec2   f32cv = f32vec2(-0.25F, 0.03f);

    f32vec2 f32v;
    f32v.x  = f32c;
    f32v   += f32cv;
}

// Block memory layout
struct S
{
    float32_t  x;
    f32vec2    y;
    f32vec3    z;
};

layout(column_major, std140) uniform B1
{
    float32_t  a;
    f32vec2    b;
    f32vec3    c;
    float32_t  d[2];
    f32mat2x3  e;
    f32mat2x3  f[2];
    S          g;
    S          h[2];
};

// Specialization constant
layout(constant_id = 100) const float16_t sf16 = 0.125hf;
layout(constant_id = 101) const float32_t sf   = 0.25;
layout(constant_id = 102) const float64_t sd   = 0.5lf;

const float  f16_to_f = float(sf16);
const double f16_to_d = float(sf16);

const float16_t f_to_f16 = float16_t(sf);
const float16_t d_to_f16 = float16_t(sd);

void operators()
{
    float32_t f32;
    f32vec2   f32v;
    f32mat2x2 f32m;
    bool      b;

    // Arithmetic
    f32v += f32v;
    f32v -= f32v;
    f32v *= f32v;
    f32v /= f32v;
    f32v++;
    f32v--;
    ++f32m;
    --f32m;
    f32v = -f32v;
    f32m = -f32m;

    f32 = f32v.x + f32v.y;
    f32 = f32v.x - f32v.y;
    f32 = f32v.x * f32v.y;
    f32 = f32v.x / f32v.y;

    // Relational
    b = (f32v.x != f32);
    b = (f32v.y == f32);
    b = (f32v.x >  f32);
    b = (f32v.y <  f32);
    b = (f32v.x >= f32);
    b = (f32v.y <= f32);

    // Vector/matrix operations
    f32v = f32v * f32;
    f32m = f32m * f32;
    f32v = f32m * f32v;
    f32v = f32v * f32m;
    f32m = f32m * f32m;
}

void typeCast()
{
    bvec3   bv;
    f32vec3   f32v;
    f64vec3   f64v;
    i8vec3    i8v;
    u8vec3    u8v;
    i16vec3   i16v;
    u16vec3   u16v;
    i32vec3   i32v;
    u32vec3   u32v;
    i64vec3   i64v;
    u64vec3   u64v;
    f16vec3   f16v;

    f64v = f32v;            // float32_t -> float64_t

    f32v = f32vec3(bv);     // bool -> float32
    bv   = bvec3(f32v);     // float32 -> bool

    f32v = f32vec3(f64v);   // double -> float32
    f64v = f64vec3(f32v);   // float32 -> double

    f32v = f32vec3(f16v);   // float16 -> float32
    f16v = f16vec3(f32v);   // float32 -> float16

    i8v  = i8vec3(f32v);    //  float32 -> int8
    i16v = i16vec3(f32v);    // float32 -> int16
    i32v = i32vec3(f32v);    // float32 -> int32
    i64v = i64vec3(f32v);    // float32 -> int64

    u8v  = u8vec3(f32v);    //  float32 -> uint8
    u16v = u16vec3(f32v);    // float32 -> uint16
    u32v = u32vec3(f32v);    // float32 -> uint32
    u64v = u64vec3(f32v);    // float32 -> uint64
}

void builtinAngleTrigFuncs()
{
    f32vec4 f32v1, f32v2;

    f32v2 = radians(f32v1);
    f32v2 = degrees(f32v1);
    f32v2 = sin(f32v1);
    f32v2 = cos(f32v1);
    f32v2 = tan(f32v1);
    f32v2 = asin(f32v1);
    f32v2 = acos(f32v1);
    f32v2 = atan(f32v1, f32v2);
    f32v2 = atan(f32v1);
    f32v2 = sinh(f32v1);
    f32v2 = cosh(f32v1);
    f32v2 = tanh(f32v1);
    f32v2 = asinh(f32v1);
    f32v2 = acosh(f32v1);
    f32v2 = atanh(f32v1);
}

void builtinExpFuncs()
{
    f32vec2 f32v1, f32v2;

    f32v2 = pow(f32v1, f32v2);
    f32v2 = exp(f32v1);
    f32v2 = log(f32v1);
    f32v2 = exp2(f32v1);
    f32v2 = log2(f32v1);
    f32v2 = sqrt(f32v1);
    f32v2 = inversesqrt(f32v1);
}

void builtinCommonFuncs()
{
    f32vec3   f32v1, f32v2, f32v3;
    float32_t f32;
    bool  b;
    bvec3 bv;
    ivec3 iv;

    f32v2 = abs(f32v1);
    f32v2 = sign(f32v1);
    f32v2 = floor(f32v1);
    f32v2 = trunc(f32v1);
    f32v2 = round(f32v1);
    f32v2 = roundEven(f32v1);
    f32v2 = ceil(f32v1);
    f32v2 = fract(f32v1);
    f32v2 = mod(f32v1, f32v2);
    f32v2 = mod(f32v1, f32);
    f32v3 = modf(f32v1, f32v2);
    f32v3 = min(f32v1, f32v2);
    f32v3 = min(f32v1, f32);
    f32v3 = max(f32v1, f32v2);
    f32v3 = max(f32v1, f32);
    f32v3 = clamp(f32v1, f32, f32v2.x);
    f32v3 = clamp(f32v1, f32v2, f32vec3(f32));
    f32v3 = mix(f32v1, f32v2, f32);
    f32v3 = mix(f32v1, f32v2, f32v3);
    f32v3 = mix(f32v1, f32v2, bv);
    f32v3 = step(f32v1, f32v2);
    f32v3 = step(f32, f32v3);
    f32v3 = smoothstep(f32v1, f32v2, f32v3);
    f32v3 = smoothstep(f32, f32v1.x, f32v2);
    b     = isnan(f32);
    bv    = isinf(f32v1);
    f32v3 = fma(f32v1, f32v2, f32v3);
    f32v2 = frexp(f32v1, iv);
    f32v2 = ldexp(f32v1, iv);
}

void builtinGeometryFuncs()
{
    float32_t f32;
    f32vec3   f32v1, f32v2, f32v3;

    f32   = length(f32v1);
    f32   = distance(f32v1, f32v2);
    f32   = dot(f32v1, f32v2);
    f32v3 = cross(f32v1, f32v2);
    f32v2 = normalize(f32v1);
    f32v3 = faceforward(f32v1, f32v2, f32v3);
    f32v3 = reflect(f32v1, f32v2);
    f32v3 = refract(f32v1, f32v2, f32);
}

void builtinMatrixFuncs()
{
    f32mat2x3 f32m1, f32m2, f32m3;
    f32mat3x2 f32m4;
    f32mat3   f32m5;
    f32mat4   f32m6, f32m7;

    f32vec3 f32v1;
    f32vec2 f32v2;

    float32_t f32;

    f32m3 = matrixCompMult(f32m1, f32m2);
    f32m1 = outerProduct(f32v1, f32v2);
    f32m4 = transpose(f32m1);
    f32   = determinant(f32m5);
    f32m6 = inverse(f32m7);
}

void builtinVecRelFuncs()
{
    f32vec3 f32v1, f32v2;
    bvec3   bv;

    bv = lessThan(f32v1, f32v2);
    bv = lessThanEqual(f32v1, f32v2);
    bv = greaterThan(f32v1, f32v2);
    bv = greaterThanEqual(f32v1, f32v2);
    bv = equal(f32v1, f32v2);
    bv = notEqual(f32v1, f32v2);
}

in f32vec3 if32v;

void builtinFragProcFuncs()
{
    f32vec3 f32v;

    // Derivative
    f32v.x  = dFdx(if32v.x);
    f32v.y  = dFdy(if32v.y);
    f32v.xy = dFdxFine(if32v.xy);
    f32v.xy = dFdyFine(if32v.xy);
    f32v    = dFdxCoarse(if32v);
    f32v    = dFdxCoarse(if32v);

    f32v.x  = fwidth(if32v.x);
    f32v.xy = fwidthFine(if32v.xy);
    f32v    = fwidthCoarse(if32v);

    // Interpolation
    f32v.x  = interpolateAtCentroid(if32v.x);
    f32v.xy = interpolateAtSample(if32v.xy, 1);
    f32v    = interpolateAtOffset(if32v, f32vec2(0.5f));
}

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
    const float64_t f64c  = 0.000001LF;
    const f64vec2   f64cv = f64vec2(-0.25lF, 0.03Lf);

    f64vec2 f64v;
    f64v.x  = f64c;
    f64v   += f64cv;
}

// Block memory layout
struct S
{
    float64_t  x;
    f64vec2    y;
    f64vec3    z;
};

layout(column_major, std140) uniform B1
{
    float64_t  a;
    f64vec2    b;
    f64vec3    c;
    float64_t  d[2];
    f64mat2x3  e;
    f64mat2x3  f[2];
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
    float64_t f64;
    f64vec2   f64v;
    f64mat2x2 f64m;
    bool      b;

    // Arithmetic
    f64v += f64v;
    f64v -= f64v;
    f64v *= f64v;
    f64v /= f64v;
    f64v++;
    f64v--;
    ++f64m;
    --f64m;
    f64v = -f64v;
    f64m = -f64m;

    f64 = f64v.x + f64v.y;
    f64 = f64v.x - f64v.y;
    f64 = f64v.x * f64v.y;
    f64 = f64v.x / f64v.y;

    // Relational
    b = (f64v.x != f64);
    b = (f64v.y == f64);
    b = (f64v.x >  f64);
    b = (f64v.y <  f64);
    b = (f64v.x >= f64);
    b = (f64v.y <= f64);

    // Vector/matrix operations
    f64v = f64v * f64;
    f64m = f64m * f64;
    f64v = f64m * f64v;
    f64v = f64v * f64m;
    f64m = f64m * f64m;
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

    f64v = f64vec3(bv);     // bool -> float64
    bv   = bvec3(f64v);     // float64 -> bool

    f64v = f64vec3(f16v);   // float16 -> float64
    f16v = f16vec3(f64v);   // float64 -> float16

    i8v  = i8vec3(f64v);    //  float64 -> int8
    i16v = i16vec3(f64v);    // float64 -> int16
    i32v = i32vec3(f64v);    // float64 -> int32
    i64v = i64vec3(f64v);    // float64 -> int64

    u8v  = u8vec3(f64v);    //  float64 -> uint8
    u16v = u16vec3(f64v);    // float64 -> uint16
    u32v = u32vec3(f64v);    // float64 -> uint32
    u64v = u64vec3(f64v);    // float64 -> uint64
}

void builtinAngleTrigFuncs()
{
    f64vec4 f64v1, f64v2;

    f64v2 = radians(f64v1);
    f64v2 = degrees(f64v1);
    f64v2 = sin(f64v1);
    f64v2 = cos(f64v1);
    f64v2 = tan(f64v1);
    f64v2 = asin(f64v1);
    f64v2 = acos(f64v1);
    f64v2 = atan(f64v1, f64v2);
    f64v2 = atan(f64v1);
    f64v2 = sinh(f64v1);
    f64v2 = cosh(f64v1);
    f64v2 = tanh(f64v1);
    f64v2 = asinh(f64v1);
    f64v2 = acosh(f64v1);
    f64v2 = atanh(f64v1);
}

void builtinExpFuncs()
{
    f64vec2 f64v1, f64v2;

    f64v2 = pow(f64v1, f64v2);
    f64v2 = exp(f64v1);
    f64v2 = log(f64v1);
    f64v2 = exp2(f64v1);
    f64v2 = log2(f64v1);
    f64v2 = sqrt(f64v1);
    f64v2 = inversesqrt(f64v1);
}

void builtinCommonFuncs()
{
    f64vec3   f64v1, f64v2, f64v3;
    float64_t f64;
    bool  b;
    bvec3 bv;
    ivec3 iv;

    f64v2 = abs(f64v1);
    f64v2 = sign(f64v1);
    f64v2 = floor(f64v1);
    f64v2 = trunc(f64v1);
    f64v2 = round(f64v1);
    f64v2 = roundEven(f64v1);
    f64v2 = ceil(f64v1);
    f64v2 = fract(f64v1);
    f64v2 = mod(f64v1, f64v2);
    f64v2 = mod(f64v1, f64);
    f64v3 = modf(f64v1, f64v2);
    f64v3 = min(f64v1, f64v2);
    f64v3 = min(f64v1, f64);
    f64v3 = max(f64v1, f64v2);
    f64v3 = max(f64v1, f64);
    f64v3 = clamp(f64v1, f64, f64v2.x);
    f64v3 = clamp(f64v1, f64v2, f64vec3(f64));
    f64v3 = mix(f64v1, f64v2, f64);
    f64v3 = mix(f64v1, f64v2, f64v3);
    f64v3 = mix(f64v1, f64v2, bv);
    f64v3 = step(f64v1, f64v2);
    f64v3 = step(f64, f64v3);
    f64v3 = smoothstep(f64v1, f64v2, f64v3);
    f64v3 = smoothstep(f64, f64v1.x, f64v2);
    b     = isnan(f64);
    bv    = isinf(f64v1);
    f64v3 = fma(f64v1, f64v2, f64v3);
    f64v2 = frexp(f64v1, iv);
    f64v2 = ldexp(f64v1, iv);
}

void builtinGeometryFuncs()
{
    float64_t f64;
    f64vec3   f64v1, f64v2, f64v3;

    f64   = length(f64v1);
    f64   = distance(f64v1, f64v2);
    f64   = dot(f64v1, f64v2);
    f64v3 = cross(f64v1, f64v2);
    f64v2 = normalize(f64v1);
    f64v3 = faceforward(f64v1, f64v2, f64v3);
    f64v3 = reflect(f64v1, f64v2);
    f64v3 = refract(f64v1, f64v2, f64);
}

void builtinMatrixFuncs()
{
    f64mat2x3 f64m1, f64m2, f64m3;
    f64mat3x2 f64m4;
    f64mat3   f64m5;
    f64mat4   f64m6, f64m7;

    f64vec3 f64v1;
    f64vec2 f64v2;

    float64_t f64;

    f64m3 = matrixCompMult(f64m1, f64m2);
    f64m1 = outerProduct(f64v1, f64v2);
    f64m4 = transpose(f64m1);
    f64   = determinant(f64m5);
    f64m6 = inverse(f64m7);
}

void builtinVecRelFuncs()
{
    f64vec3 f64v1, f64v2;
    bvec3   bv;

    bv = lessThan(f64v1, f64v2);
    bv = lessThanEqual(f64v1, f64v2);
    bv = greaterThan(f64v1, f64v2);
    bv = greaterThanEqual(f64v1, f64v2);
    bv = equal(f64v1, f64v2);
    bv = notEqual(f64v1, f64v2);
}

in flat f64vec3 if64v;

void builtinFragProcFuncs()
{
    f64vec3 f64v;

    // Derivative
    f64v.x  = dFdx(if64v.x);
    f64v.y  = dFdy(if64v.y);
    f64v.xy = dFdxFine(if64v.xy);
    f64v.xy = dFdyFine(if64v.xy);
    f64v    = dFdxCoarse(if64v);
    f64v    = dFdxCoarse(if64v);

    f64v.x  = fwidth(if64v.x);
    f64v.xy = fwidthFine(if64v.xy);
    f64v    = fwidthCoarse(if64v);

    // Interpolation
    f64v.x  = interpolateAtCentroid(if64v.x);
    f64v.xy = interpolateAtSample(if64v.xy, 1);
    f64v    = interpolateAtOffset(if64v, f64vec2(0.5f));
}

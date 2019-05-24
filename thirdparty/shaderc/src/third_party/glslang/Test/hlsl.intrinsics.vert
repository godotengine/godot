float VertexShaderFunctionS(float inF0, float inF1, float inF2, uint inU0, uint inU1)
{
    all(inF0);
    abs(inF0);
    acos(inF0);
    any(inF0);
    asin(inF0);
    asint(inF0);
    asuint(inF0);
    asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    atan(inF0);
    atan2(inF0, inF1);
    ceil(inF0);
    clamp(inF0, inF1, inF2);
    cos(inF0);
    cosh(inF0);
    countbits(7);
    degrees(inF0);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    exp(inF0);
    exp2(inF0);
    firstbithigh(7);
    firstbitlow(7);
    floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    fmod(inF0, inF1);
    frac(inF0);
    isinf(inF0);
    isnan(inF0);
    ldexp(inF0, inF1);
    lerp(inF0, inF1, inF2);
    log(inF0);
    log10(inF0);
    log2(inF0);
    max(inF0, inF1);
    min(inF0, inF1);
    // TODO: mul(inF0, inF1);
    pow(inF0, inF1);
    radians(inF0);
    reversebits(2);
    round(inF0);
    rsqrt(inF0);
    saturate(inF0);
    sign(inF0);
    sin(inF0);
    sincos(inF0, inF1, inF2);
    sinh(inF0);
    smoothstep(inF0, inF1, inF2);
    sqrt(inF0);
    step(inF0, inF1);
    tan(inF0);
    tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    trunc(inF0);

    return 0.0;
}

float1 VertexShaderFunction1(float1 inF0, float1 inF1, float1 inF2)
{
    // TODO: ... add when float1 prototypes are generated
    return 0.0;
}

float2 VertexShaderFunction2(float2 inF0, float2 inF1, float2 inF2, uint2 inU0, uint2 inU1)
{
    all(inF0);
    abs(inF0);
    acos(inF0);
    any(inF0);
    asin(inF0);
    asint(inF0);
    asuint(inF0);
    asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    atan(inF0);
    atan2(inF0, inF1);
    ceil(inF0);
    clamp(inF0, inF1, inF2);
    cos(inF0);
    cosh(inF0);
    countbits(int2(7,3));
    degrees(inF0);
    distance(inF0, inF1);
    dot(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    exp(inF0);
    exp2(inF0);
    faceforward(inF0, inF1, inF2);
    firstbithigh(7);
    firstbitlow(7);
    floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    fmod(inF0, inF1);
    frac(inF0);
    isinf(inF0);
    isnan(inF0);
    ldexp(inF0, inF1);
    lerp(inF0, inF1, inF2);
    length(inF0);
    log(inF0);
    log10(inF0);
    log2(inF0);
    max(inF0, inF1);
    min(inF0, inF1);
    // TODO: mul(inF0, inF1);
    normalize(inF0);
    pow(inF0, inF1);
    radians(inF0);
    reflect(inF0, inF1);
    refract(inF0, inF1, 2.0);
    reversebits(int2(1,2));
    round(inF0);
    rsqrt(inF0);
    saturate(inF0);
    sign(inF0);
    sin(inF0);
    sincos(inF0, inF1, inF2);
    sinh(inF0);
    smoothstep(inF0, inF1, inF2);
    sqrt(inF0);
    step(inF0, inF1);
    tan(inF0);
    tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float2(1,2);
}

float3 VertexShaderFunction3(float3 inF0, float3 inF1, float3 inF2, uint3 inU0, uint3 inU1)
{
    all(inF0);
    abs(inF0);
    acos(inF0);
    any(inF0);
    asin(inF0);
    asint(inF0);
    asuint(inF0);
    asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    atan(inF0);
    atan2(inF0, inF1);
    ceil(inF0);
    clamp(inF0, inF1, inF2);
    cos(inF0);
    cosh(inF0);
    countbits(int3(7,3,5));
    cross(inF0, inF1);
    degrees(inF0);
    distance(inF0, inF1);
    dot(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    exp(inF0);
    exp2(inF0);
    faceforward(inF0, inF1, inF2);
    firstbithigh(7);
    firstbitlow(7);
    floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    fmod(inF0, inF1);
    frac(inF0);
    isinf(inF0);
    isnan(inF0);
    ldexp(inF0, inF1);
    lerp(inF0, inF1, inF2);
    length(inF0);
    log(inF0);
    log10(inF0);
    log2(inF0);
    max(inF0, inF1);
    min(inF0, inF1);
    // TODO: mul(inF0, inF1);
    normalize(inF0);
    pow(inF0, inF1);
    radians(inF0);
    reflect(inF0, inF1);
    refract(inF0, inF1, 2.0);
    reversebits(int3(1,2,3));
    round(inF0);
    rsqrt(inF0);
    saturate(inF0);
    sign(inF0);
    sin(inF0);
    sincos(inF0, inF1, inF2);
    sinh(inF0);
    smoothstep(inF0, inF1, inF2);
    sqrt(inF0);
    step(inF0, inF1);
    tan(inF0);
    tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float3(1,2,3);
}

float4 VertexShaderFunction4(float4 inF0, float4 inF1, float4 inF2, uint4 inU0, uint4 inU1)
{
    all(inF0);
    abs(inF0);
    acos(inF0);
    any(inF0);
    asin(inF0);
    asint(inF0);
    asuint(inF0);
    asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    atan(inF0);
    atan2(inF0, inF1);
    ceil(inF0);
    clamp(inF0, inF1, inF2);
    cos(inF0);
    cosh(inF0);
    countbits(int4(7,3,5,2));
    degrees(inF0);
    distance(inF0, inF1);
    dot(inF0, inF1);
    dst(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    exp(inF0);
    exp2(inF0);
    faceforward(inF0, inF1, inF2);
    firstbithigh(7);
    firstbitlow(7);
    floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    fmod(inF0, inF1);
    frac(inF0);
    isinf(inF0);
    isnan(inF0);
    ldexp(inF0, inF1);
    lerp(inF0, inF1, inF2);
    length(inF0);
    log(inF0);
    log10(inF0);
    log2(inF0);
    max(inF0, inF1);
    min(inF0, inF1);
    // TODO: mul(inF0, inF1);
    normalize(inF0);
    pow(inF0, inF1);
    radians(inF0);
    reflect(inF0, inF1);
    refract(inF0, inF1, 2.0);
    reversebits(int4(1,2,3,4));
    round(inF0);
    rsqrt(inF0);
    saturate(inF0);
    sign(inF0);
    sin(inF0);
    sincos(inF0, inF1, inF2);
    sinh(inF0);
    smoothstep(inF0, inF1, inF2);
    sqrt(inF0);
    step(inF0, inF1);
    tan(inF0);
    tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float4(1,2,3,4);
}

// TODO: for mats:
//    asfloat(inU0); \
//    asint(inF0); \
//    asuint(inF0); \

// TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
#define MATFNS() \
    all(inF0); \
    abs(inF0); \
    acos(inF0); \
    any(inF0); \
    asin(inF0); \
    atan(inF0); \
    atan2(inF0, inF1); \
    ceil(inF0); \
    clamp(inF0, inF1, inF2); \
    cos(inF0); \
    cosh(inF0); \
    degrees(inF0); \
    determinant(inF0); \
    exp(inF0); \
    exp2(inF0); \
    firstbithigh(7); \
    firstbitlow(7); \
    floor(inF0); \
    fmod(inF0, inF1); \
    frac(inF0); \
    ldexp(inF0, inF1); \
    lerp(inF0, inF1, inF2); \
    log(inF0); \
    log10(inF0); \
    log2(inF0); \
    max(inF0, inF1); \
    min(inF0, inF1); \
    pow(inF0, inF1); \
    radians(inF0); \
    round(inF0); \
    rsqrt(inF0); \
    saturate(inF0); \
    sign(inF0); \
    sin(inF0); \
    sincos(inF0, inF1, inF2); \
    sinh(inF0); \
    smoothstep(inF0, inF1, inF2); \
    sqrt(inF0); \
    step(inF0, inF1); \
    tan(inF0); \
    tanh(inF0); \
    transpose(inF0); \
    trunc(inF0);

// TODO: turn on non-square matrix tests when protos are available.

float2x2 VertexShaderFunction2x2(float2x2 inF0, float2x2 inF1, float2x2 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS();

    // TODO: ... add when float1 prototypes are generated
    return float2x2(2,2,2,2);
}

float3x3 VertexShaderFunction3x3(float3x3 inF0, float3x3 inF1, float3x3 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS();

    // TODO: ... add when float1 prototypes are generated
    return float3x3(3,3,3,3,3,3,3,3,3);
}

float4x4 VertexShaderFunction4x4(float4x4 inF0, float4x4 inF1, float4x4 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS();

    // TODO: ... add when float1 prototypes are generated
    return float4x4(4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4);
}

#define TESTGENMUL(ST, VT, MT) \
    ST r0 = mul(inF0,  inF1);  \
    VT r1 = mul(inFV0, inF0);  \
    VT r2 = mul(inF0,  inFV0); \
    ST r3 = mul(inFV0, inFV1); \
    VT r4 = mul(inFM0, inFV0); \
    VT r5 = mul(inFV0, inFM0); \
    MT r6 = mul(inFM0, inF0);  \
    MT r7 = mul(inF0, inFM0);  \
    MT r8 = mul(inFM0, inFM1);


void TestGenMul2(float inF0, float inF1,
                 float2 inFV0, float2 inFV1,
                 float2x2 inFM0, float2x2 inFM1)
{
    TESTGENMUL(float, float2, float2x2);
}

void TestGenMul3(float inF0, float inF1,
                 float3 inFV0, float3 inFV1,
                 float3x3 inFM0, float3x3 inFM1)
{
    TESTGENMUL(float, float3, float3x3);
}

void TestGenMul4(float inF0, float inF1,
                 float4 inFV0, float4 inFV1,
                 float4x4 inFM0, float4x4 inFM1)
{
    TESTGENMUL(float, float4, float4x4);
}

// Test some non-square mats
void TestGenMulNxM(float inF0, float inF1,
                   float2 inFV2, float3 inFV3,
                   float2x3 inFM2x3, float3x2 inFM3x2,
                   float3x3 inFM3x3, float3x4 inFM3x4,
                   float2x4 inFM2x4)
{
    float  r00 = mul(inF0,  inF1);  // S=S*S
    float2 r01 = mul(inFV2, inF0);  // V=V*S
    float3 r02 = mul(inFV3, inF0);  // V=V*S
    float2 r03 = mul(inF0,  inFV2); // V=S*V
    float3 r04 = mul(inF0,  inFV3); // V=S*V
    float  r05 = mul(inFV2, inFV2); // S=V*V
    float  r06 = mul(inFV3, inFV3); // S=V*V
    float3 r07 = mul(inFV2, inFM2x3); // V=V*M (return V dim is Mcols)
    float2 r08 = mul(inFV3, inFM3x2); // V=V*M (return V dim is Mcols)
    float2 r09 = mul(inFM2x3, inFV3); // V=M*V (return V dim is Mrows)
    float3 r10 = mul(inFM3x2, inFV2); // V=M*V (return V dim is Mrows)
    float2x3 r11 = mul(inFM2x3, inF0);
    float3x2 r12 = mul(inFM3x2, inF0);
    float2x2 r13 = mul(inFM2x3, inFM3x2);
    float2x3 r14 = mul(inFM2x3, inFM3x3);
    float2x4 r15 = mul(inFM2x3, inFM3x4);
    float3x4 r16 = mul(inFM3x2, inFM2x4);
}

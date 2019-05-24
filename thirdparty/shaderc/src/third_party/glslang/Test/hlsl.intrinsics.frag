

groupshared uint gs_ua;
groupshared uint gs_ub;
groupshared uint gs_uc;
groupshared uint2 gs_ua2;
groupshared uint2 gs_ub2;
groupshared uint2 gs_uc2;
groupshared uint3 gs_ua3;
groupshared uint3 gs_ub3;
groupshared uint3 gs_uc3;
groupshared uint4 gs_ua4;
groupshared uint4 gs_ub4;
groupshared uint4 gs_uc4;

float PixelShaderFunctionS(float inF0, float inF1, float inF2, uint inU0, int inU1)
{
    uint out_u1;

    bool r000 = all(inF0);
    float r001 = abs(inF0);
    float r002 = acos(inF0);
    bool r003 = any(inF0);
    float r004 = asin(inF0);
    int r005 = asint(inF0);
    uint r006 = asuint(inU1);
    float r007 = asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    float r009 = atan(inF0);
    float r010 = atan2(inF0, inF1);
    float r011 = ceil(inF0);
    float r012 = clamp(inF0, inF1, inF2);
    clip(inF0);
    clip(r005);
    float r014 = cos(inF0);
    float r015 = cosh(inF0);
    int r016 = countbits(7);
    float r017 = ddx(inF0);
    float r018 = ddx_coarse(inF0);
    float r019 = ddx_fine(inF0);
    float r020 = ddy(inF0);
    float r021 = ddy_coarse(inF0);
    float r022 = ddy_fine(inF0);
    float r023 = degrees(inF0);
    float r024 = distance(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    float r027 = exp(inF0);
    float r028 = exp2(inF0);
    uint r029 = firstbithigh(7);
    uint r030 = firstbitlow(7);
    float r031 = floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    float r033 = fmod(inF0, inF1);
    float r034 = frac(inF0);
    float r036 = fwidth(inF0);
    bool r037 = isinf(inF0);
    bool r038 = isnan(inF0);
    float r039 = ldexp(inF0, inF1);
    float r039a = lerp(inF0, inF1, inF2);
    float r040 = log(inF0);
    float r041 = log10(inF0);
    float r042 = log2(inF0);
    float r043 = max(inF0, inF1);
    float r044 = min(inF0, inF1);
    float r045 = pow(inF0, inF1);
    float r046 = radians(inF0);
    float r047 = rcp(inF0);
    uint r048 = reversebits(2);
    float r049 = round(inF0);
    float r050 = rsqrt(inF0);
    float r051 = saturate(inF0);
    float r052 = sign(inF0);
    float r053 = sin(inF0);
    sincos(inF0, inF1, inF2);
    float r055 = sinh(inF0);
    float r056 = smoothstep(inF0, inF1, inF2);
    float r057 = sqrt(inF0);
    float r058 = step(inF0, inF1);
    float r059 = tan(inF0);
    float r060 = tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    float r061 = trunc(inF0);

    return 0.0;
}

float1 PixelShaderFunction1(float1 inF0, float1 inF1, float1 inF2)
{
    // TODO: ... add when float1 prototypes are generated
    return 0.0;
}

float2 PixelShaderFunction2(float2 inF0, float2 inF1, float2 inF2, uint2 inU0, uint2 inU1)
{
    uint2 out_u2;

    bool r000 = all(inF0);
    float2 r001 = abs(inF0);
    float2 r002 = acos(inF0);
    bool r003 = any(inF0);
    float2 r004 = asin(inF0);
    int2 r005 = asint(inF0);
    uint2 r006 = asuint(inF0);
    float2 r007 = asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    float2 r009 = atan(inF0);
    float2 r010 = atan2(inF0, inF1);
    float2 r011 = ceil(inF0);
    float2 r012 = clamp(inF0, inF1, inF2);
    clip(inF0);
    clip(inU0);
    float2 r013 = cos(inF0);
    float2 r015 = cosh(inF0);
    int2 r016 = countbits(int2(7,3));
    float2 r017 = ddx(inF0);
    float2 r018 = ddx_coarse(inF0);
    float2 r019 = ddx_fine(inF0);
    float2 r020 = ddy(inF0);
    float2 r021 = ddy_coarse(inF0);
    float2 r022 = ddy_fine(inF0);
    float2 r023 = degrees(inF0);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    float r026 = distance(inF0, inF1);
    float r027 = dot(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    float2 r028 = exp(inF0);
    float2 r029 = exp2(inF0);
    float2 r030 = faceforward(inF0, inF1, inF2);
    uint2 r031 = firstbithigh(uint2(7,8));
    uint2 r032 = firstbitlow(uint2(7,8));
    float2 r033 = floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    float2 r035 = fmod(inF0, inF1);
    float2 r036 = frac(inF0);
    float2 r038 = fwidth(inF0);
    bool2 r039 = isinf(inF0);
    bool2 r040 = isnan(inF0);
    float2 r041 = ldexp(inF0, inF1);
    float2 r039a = lerp(inF0, inF1, inF2);
    float r042 = length(inF0);
    float2 r043 = log(inF0);
    float2 r044 = log10(inF0);
    float2 r045 = log2(inF0);
    float2 r046 = max(inF0, inF1);
    float2 r047 = min(inF0, inF1);
    float2 r048 = normalize(inF0);
    float2 r049 = pow(inF0, inF1);
    float2 r050 = radians(inF0);
    float2 r051 = rcp(inF0);
    float2 r052 = reflect(inF0, inF1);
    float2 r053 = refract(inF0, inF1, 2.0);
    uint2 r054 = reversebits(uint2(1,2));
    float2 r055 = round(inF0);
    float2 r056 = rsqrt(inF0);
    float2 r057 = saturate(inF0);
    float2 r058 = sign(inF0);
    float2 r059 = sin(inF0);
    sincos(inF0, inF1, inF2);
    float2 r060 = sinh(inF0);
    float2 r061 = smoothstep(inF0, inF1, inF2);
    float2 r062 = sqrt(inF0);
    float2 r063 = step(inF0, inF1);
    float2 r064 = tan(inF0);
    float2 r065 = tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    float2 r066 = trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float2(1,2);
}

float3 PixelShaderFunction3(float3 inF0, float3 inF1, float3 inF2, uint3 inU0, uint3 inU1)
{
    uint3 out_u3;
    
    bool r000 = all(inF0);
    float3 r001 = abs(inF0);
    float3 r002 = acos(inF0);
    bool r003 = any(inF0);
    float3 r004 = asin(inF0);
    int3 r005 = asint(inF0);
    uint3 r006 = asuint(inF0);
    float3 r007 = asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    float3 r009 = atan(inF0);
    float3 r010 = atan2(inF0, inF1);
    float3 r011 = ceil(inF0);
    float3 r012 = clamp(inF0, inF1, inF2);
    clip(inF0);
    clip(inU0);
    float3 r013 = cos(inF0);
    float3 r014 = cosh(inF0);
    uint3 r015 = countbits(uint3(7,3,5));
    float3 r016 = cross(inF0, inF1);
    float3 r017 = ddx(inF0);
    float3 r018 = ddx_coarse(inF0);
    float3 r019 = ddx_fine(inF0);
    float3 r020 = ddy(inF0);
    float3 r021 = ddy_coarse(inF0);
    float3 r022 = ddy_fine(inF0);
    float3 r023 = degrees(inF0);
    float r024 = distance(inF0, inF1);
    float r025 = dot(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    float3 r029 = exp(inF0);
    float3 r030 = exp2(inF0);
    float3 r031 = faceforward(inF0, inF1, inF2);
    uint3 r032 = firstbithigh(uint3(2,3,4));
    uint3 r033 = firstbitlow(uint3(2,3,4));
    float3 r034 = floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    float3 r036 = fmod(inF0, inF1);
    float3 r037 = frac(inF0);
    float3 r039 = fwidth(inF0);
    bool3 r040 = isinf(inF0);
    bool3 r041 = isnan(inF0);
    float3 r042 = ldexp(inF0, inF1);
    float3 r039a = lerp(inF0, inF1, inF2);
    float3 r039b = lerp(inF0, inF1, 0.3); // test vec,vec,scalar lerp
    float r043 = length(inF0);
    float3 r044 = log(inF0);
    float3 r045 = log10(inF0);
    float3 r046 = log2(inF0);
    float3 r047 = max(inF0, inF1);
    float3 r048 = min(inF0, inF1);
    float3 r049 = normalize(inF0);
    float3 r050 = pow(inF0, inF1);
    float3 r051 = radians(inF0);
    float3 r052 = rcp(inF0);
    float3 r053 = reflect(inF0, inF1);
    float3 r054 = refract(inF0, inF1, 2.0);
    uint3 r055 = reversebits(uint3(1,2,3));
    float3 r056 = round(inF0);
    float3 r057 = rsqrt(inF0);
    float3 r058 = saturate(inF0);
    float3 r059 = sign(inF0);
    float3 r060 = sin(inF0);
    sincos(inF0, inF1, inF2);
    float3 r061 = sinh(inF0);
    float3 r062 = smoothstep(inF0, inF1, inF2);
    float3 r063 = sqrt(inF0);
    float3 r064 = step(inF0, inF1);
    float3 r065 = tan(inF0);
    float3 r066 = tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    float3 r067 = trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float3(1,2,3);
}

float4 PixelShaderFunction(float4 inF0, float4 inF1, float4 inF2, uint4 inU0, uint4 inU1)
{
    uint4 out_u4;

    bool r000 = all(inF0);
    float4 r001 = abs(inF0);
    float4 r002 = acos(inF0);
    bool r003 = any(inF0);
    float4 r004 = asin(inF0);
    int4 r005 = asint(inF0);
    uint4 r006 = asuint(inF0);
    float4 r007 = asfloat(inU0);
    // asdouble(inU0, inU1);  // TODO: enable when HLSL parser used for intrinsics
    float4 r009 = atan(inF0);
    float4 r010 = atan2(inF0, inF1);
    float4 r011 = ceil(inF0);
    float4 r012 = clamp(inF0, inF1, inF2);
    clip(inF0);
    clip(inU0);
    float4 r013 = cos(inF0);
    float4 r014 = cosh(inF0);
    uint4 r015 = countbits(uint4(7,3,5,2));
    float4 r016 = ddx(inF0);
    float4 r017 = ddx_coarse(inF0);
    float4 r018 = ddx_fine(inF0);
    float4 r019 = ddy(inF0);
    float4 r020 = ddy_coarse(inF0);
    float4 r021 = ddy_fine(inF0);
    float4 r022 = degrees(inF0);
    float r023 = distance(inF0, inF1);
    float r024 = dot(inF0, inF1);
    float4 r025 = dst(inF0, inF1);
    // EvaluateAttributeAtCentroid(inF0);
    // EvaluateAttributeAtSample(inF0, 0);
    // TODO: EvaluateAttributeSnapped(inF0, int2(1,2));
    float4 r029 = exp(inF0);
    float4 r030 = exp2(inF0);
    float4 r031 = faceforward(inF0, inF1, inF2);
    uint4 r032 = firstbithigh(uint4(7,8,9,10));
    uint4 r033 = firstbitlow(uint4(7,8,9,10));
    float4 r034 = floor(inF0);
    // TODO: fma(inD0, inD1, inD2);
    float4 r036 = fmod(inF0, inF1);
    float4 r037 = frac(inF0);
    float4 r039 = fwidth(inF0);
    bool4 r040 = isinf(inF0);
    bool4 r041 = isnan(inF0);
    float4 r042 = ldexp(inF0, inF1);
    float4 r039a = lerp(inF0, inF1, inF2);
    float r043 = length(inF0);
    float4 r044 = log(inF0);
    float4 r045 = log10(inF0);
    float4 r046 = log2(inF0);
    float4 r047 = max(inF0, inF1);
    float4 r048 = min(inF0, inF1);
    float4 r049 = normalize(inF0);
    float4 r050 = pow(inF0, inF1);
    float4 r051 = radians(inF0);
    float4 r052 = rcp(inF0);
    float4 r053 = reflect(inF0, inF1);
    float4 r054 = refract(inF0, inF1, 2.0);
    uint4 r055 = reversebits(uint4(1,2,3,4));
    float4 r056 = round(inF0);
    float4 r057 = rsqrt(inF0);
    float4 r058 = saturate(inF0);
    float4 r059 = sign(inF0);
    float4 r060 = sin(inF0);
    sincos(inF0, inF1, inF2);
    float4 r061 = sinh(inF0);
    float4 r062 = smoothstep(inF0, inF1, inF2);
    float4 r063 = sqrt(inF0);
    float4 r064 = step(inF0, inF1);
    float4 r065 = tan(inF0);
    float4 r066 = tanh(inF0);
    // TODO: sampler intrinsics, when we can declare the types.
    float4 r067 = trunc(inF0);

    // TODO: ... add when float1 prototypes are generated
    return float4(1,2,3,4);
}

// TODO: for mats:
//    asfloat(inU0); \
//    asint(inF0); \
//    asuint(inF0); \

// TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
#define MATFNS(MT)                          \
    bool r000 = all(inF0);                  \
    MT r001 = abs(inF0);                    \
    acos(inF0);                             \
    bool r003 = any(inF0);                  \
    MT r004 = asin(inF0);                   \
    MT r005 = atan(inF0);                   \
    MT r006 = atan2(inF0, inF1);            \
    MT r007 = ceil(inF0);                   \
    clip(inF0);                             \
    MT r008 = clamp(inF0, inF1, inF2);      \
    MT r009 = cos(inF0);                    \
    MT r010 = cosh(inF0);                   \
    MT r011 = ddx(inF0);                    \
    MT r012 = ddx_coarse(inF0);             \
    MT r013 = ddx_fine(inF0);               \
    MT r014 = ddy(inF0);                    \
    MT r015 = ddy_coarse(inF0);             \
    MT r016 = ddy_fine(inF0);               \
    MT r017 = degrees(inF0);                \
    float r018 = determinant(inF0);         \
    MT r019 = exp(inF0);                    \
    MT R020 = exp2(inF0);                   \
    MT r021 = floor(inF0);                  \
    MT r022 = fmod(inF0, inF1);             \
    MT r023 = frac(inF0);                   \
    MT r025 = fwidth(inF0);                 \
    MT r026 = ldexp(inF0, inF1);            \
    MT r026a = lerp(inF0, inF1, inF2);      \
    MT r027 = log(inF0);                    \
    MT r028 = log10(inF0);                  \
    MT r029 = log2(inF0);                   \
    MT r030 = max(inF0, inF1);              \
    MT r031 = min(inF0, inF1);              \
    MT r032 = pow(inF0, inF1);              \
    MT r033 = radians(inF0);                \
    MT r034 = round(inF0);                  \
    MT r035 = rsqrt(inF0);                  \
    MT r036 = saturate(inF0);               \
    MT r037 = sign(inF0);                   \
    MT r038 = sin(inF0);                    \
    sincos(inF0, inF1, inF2);               \
    MT r039 = sinh(inF0);                   \
    MT r049 = smoothstep(inF0, inF1, inF2); \
    MT r041 = sqrt(inF0);                   \
    MT r042 = step(inF0, inF1);             \
    MT r043 = tan(inF0);                    \
    MT r044 = tanh(inF0);                   \
    transpose(inF0);                        \
    MT r046 = trunc(inF0);

// TODO: turn on non-square matrix tests when protos are available.

float2x2 PixelShaderFunction2x2(float2x2 inF0, float2x2 inF1, float2x2 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS(float2x2);

    // TODO: ... add when float1 prototypes are generated
    return float2x2(2,2,2,2);
}

float3x3 PixelShaderFunction3x3(float3x3 inF0, float3x3 inF1, float3x3 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS(float3x3);

    // TODO: ... add when float1 prototypes are generated
    return float3x3(3,3,3,3,3,3,3,3,3);
}

float4x4 PixelShaderFunction4x4(float4x4 inF0, float4x4 inF1, float4x4 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS(float4x4);

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

struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
};

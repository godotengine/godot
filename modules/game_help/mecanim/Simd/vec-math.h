#pragma once

#include "vec-intrin.h"
#include "vec-scalar.h"

#if defined(_MSC_VER)
#   include <cstdlib>
#endif

#if !defined(MATH_HAS_SIMD_FLOAT) || !defined(__GNUC__) || !defined(__OPTIMIZE__)
#   include <cmath>
#endif

#include "vec-types.h"
#include "vec-const.h"

namespace math
{
#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 as_int4(const float4 &x)
    {
#   if defined(__ARM_NEON)
        return vreinterpretq_s32_f32((float32x4_t)x);
#   elif defined(__SSE2__)
        return _mm_castps_si128((__m128)x);
#   else
        return ((const int4*)&x)[0];
#   endif
    }

    static MATH_FORCEINLINE int3 as_int3(const float3 &x)
    {
        return as_int4(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE int2 as_int2(const float2 &x)
    {
        return as_int4(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float4 as_float4(const int4 &x)
    {
#   if defined(__ARM_NEON)
        return vreinterpretq_f32_s32((int32x4_t)x);
#   elif defined(__SSE2__)
        return _mm_castsi128_ps((__m128i)x);
#   else
        return ((const float4*)&x)[0];
#   endif
    }

    static MATH_FORCEINLINE float3 as_float3(const int3 &x)
    {
        return as_float4(int4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 as_float2(const int2 &x)
    {
        return as_float4(int4(as_native(x))).xy;
    }

#else

    static MATH_FORCEINLINE int4 as_int4(const float4 &x)
    {
        return ((const int4*)&x)[0];
    }

    static MATH_FORCEINLINE int3 as_int3(const float3 &x)
    {
        return ((const int3*)&x)[0];
    }

    static MATH_FORCEINLINE int2 as_int2(const float2 &x)
    {
        return ((const int2*)&x)[0];
    }

    static MATH_FORCEINLINE float4 as_float4(const int4 &x)
    {
        return ((const float4*)&x)[0];
    }

    static MATH_FORCEINLINE float3 as_float3(const int3 &x)
    {
        return ((const float3*)&x)[0];
    }

    static MATH_FORCEINLINE float2 as_float2(const int2 &x)
    {
        return ((const float2*)&x)[0];
    }

#endif

    // convert_intn, convert_floatn
#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 convert_int4(const float4 &v)
    {
#   if defined(__ARM_NEON)
        return vcvtq_s32_f32((float32x4_t)v);
#   elif defined(__SSE2__)
        return _mm_cvttps_epi32((__m128)v);
#   else
        return int4((int)v.x, (int)v.y, (int)v.z, (int)v.w);
#   endif
    }

    static MATH_FORCEINLINE int3 convert_int3(const float3 &v)
    {
        return convert_int4(float4(as_native(v))).xyz;
    }

    static MATH_FORCEINLINE int2 convert_int2(const float2 &v)
    {
        return convert_int4(float4(as_native(v))).xy;
    }

    static MATH_FORCEINLINE int1 convert_int1(const float1 &v)
    {
        return int1((int1::packed)convert_int4(float4((float1::packed)v)));
    }

    static MATH_FORCEINLINE float4 convert_float4(const int4 &v)
    {
#   if defined(__ARM_NEON)
        return vcvtq_f32_s32((int32x4_t)v);
#   elif defined(__SSE2__)
        return _mm_cvtepi32_ps((__m128i)v);
#   else
        return float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
#   endif
    }

    static MATH_FORCEINLINE float3 convert_float3(const int3 &v)
    {
        return convert_float4(int4(as_native(v))).xyz;
    }

    static MATH_FORCEINLINE float2 convert_float2(const int2 &v)
    {
        return convert_float4(int4(as_native(v))).xy;
    }

    static MATH_FORCEINLINE float1 convert_float1(const int1 &v)
    {
        return float1((float1::packed)convert_float4(int4((int1::packed)v)));
    }

#else

    static MATH_FORCEINLINE int4 convert_int4(const float4 &v)
    {
        return int4((int)v.x, (int)v.y, (int)v.z, (int)v.w);
    }

    static MATH_FORCEINLINE int3 convert_int3(const float3 &v)
    {
        return int3((int)v.x, (int)v.y, (int)v.z);
    }

    static MATH_FORCEINLINE int2 convert_int2(const float2 &v)
    {
        return int2((int)v.x, (int)v.y);
    }

    static MATH_FORCEINLINE int1 convert_int1(const float1 &v)
    {
        return int1((int)v);
    }

    static MATH_FORCEINLINE float4 convert_float4(const int4 &v)
    {
        return float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
    }

    static MATH_FORCEINLINE float3 convert_float3(const int3 &v)
    {
        return float3((float)v.x, (float)v.y, (float)v.z);
    }

    static MATH_FORCEINLINE float2 convert_float2(const int2 &v)
    {
        return float2((float)v.x, (float)v.y);
    }

    static MATH_FORCEINLINE float1 convert_float1(const int1 &v)
    {
        return float1((float)v);
    }

#endif

    //
    //  bitselect: bitwise select
    //  return value: (a & ~c) | (b & c)
    //
#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 bitselect(const int4 &a, const int4 &b, const int4 &c)
    {
#   if defined(__ARM_NEON)
        return vbslq_s32(vreinterpretq_u32_s32((int32x4_t)c), (int32x4_t)b, (int32x4_t)a);
#   elif defined(__SSE2__)
        return _mm_or_si128(_mm_and_si128((__m128i)c, (__m128i)b), _mm_andnot_si128((__m128i)c, (__m128i)a));
#   elif defined(__SSE__)
        return _mm_castps_si128(_mm_or_ps(_mm_and_ps(_mm_castsi128_ps((__m128i)c), _mm_castsi128_ps((__m128)b)), _mm_andnot_ps(_mm_castsi128_ps((__m128i)c), _mm_castsi128_ps((__m128)a))));
#   else
        return (a & ~c) | (b & c);
#   endif
    }

    static MATH_FORCEINLINE int bitselect(int a, int b, int c)
    {
        return (a & ~c) | (b & c);
    }

    static MATH_FORCEINLINE int3 bitselect(const int3 &a, const int3 &b, const int3 &c)
    {
        return bitselect(int4(as_native(a)), int4(as_native(b)), int4(as_native(c))).xyz;
    }

    static MATH_FORCEINLINE int2 bitselect(const int2 &a, const int2 &b, const int2 &c)
    {
        return bitselect(int4(as_native(a)), int4(as_native(b)), int4(as_native(c))).xy;
    }

    static MATH_FORCEINLINE int1 bitselect(const int1 &a, const int1 &b, const int1 &c)
    {
        return int1((int1::packed)bitselect(int4((int1::packed)a), int4((int1::packed)b), int4((int1::packed)c)));
    }

#else

    static MATH_FORCEINLINE int bitselect(int a, int b, int c)
    {
        return (a & ~c) | (b & c);
    }

    static MATH_FORCEINLINE int4 bitselect(const int4 &a, const int4 &b, const int4 &c)
    {
        return int4(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y), bitselect(a.z, b.z, c.z), bitselect(a.w, b.w, c.w));
    }

    static MATH_FORCEINLINE int3 bitselect(const int3 &a, const int3 &b, const int3 &c)
    {
        return int3(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y), bitselect(a.z, b.z, c.z));
    }

    static MATH_FORCEINLINE int2 bitselect(const int2 &a, const int2 &b, const int2 &c)
    {
        return int2(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y));
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 bitselect(const float4 &a, const float4 &b, const int4 &c)
    {
#   if defined(__ARM_NEON)
        return vbslq_f32(vreinterpretq_u32_s32((int32x4_t)c), (float32x4_t)b, (float32x4_t)a);
#   elif defined(__SSE__)
        return _mm_or_ps(_mm_and_ps(_mm_castsi128_ps((__m128i)c), (__m128)b), _mm_andnot_ps(_mm_castsi128_ps((__m128i)c), (__m128)a));
#   else
        return as_float4((as_int4(a) & ~c) | (as_int4(b) & c));
#   endif
    }

    static MATH_FORCEINLINE float bitselect(float a, float b, int c)
    {
#   if defined(__SSE__)
        return _mm_cvtss_f32((__m128)bitselect(float4(_mm_cvtf32_ss(a)), float4(_mm_cvtf32_ss(b)), int4(_mm_cvtsi32_si128(c))));
#   else
        int i = (((const int*)&a)[0] & ~c) | (((const int*)&b)[0] & c);
        return ((const float*)&i)[0];
#   endif
    }

    static MATH_FORCEINLINE float3 bitselect(const float3 &a, const float3 &b, const int3 &c)
    {
        return bitselect(float4(as_native(a)), float4(as_native(b)), int4(as_native(c))).xyz;
    }

    static MATH_FORCEINLINE float2 bitselect(const float2 &a, const float2 &b, const int2 &c)
    {
        return bitselect(float4(as_native(a)), float4(as_native(b)), int4(as_native(c))).xy;
    }

    static MATH_FORCEINLINE float1 bitselect(const float1 &a, const float1 &b, const int1 &c)
    {
        return float1((float1::packed)bitselect(float4((float1::packed)a), float4((float1::packed)b), int4((int1::packed)c)));
    }

#else

    static MATH_FORCEINLINE float bitselect(float a, float b, int c)
    {
        int i = (((const int*)&a)[0] & ~c) | (((const int*)&b)[0] & c);
        return ((const float*)&i)[0];
    }

    static MATH_FORCEINLINE float4 bitselect(const float4 &a, const float4 &b, const int4 &c)
    {
        return float4(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y), bitselect(a.z, b.z, c.z), bitselect(a.w, b.w, c.w));
    }

    static MATH_FORCEINLINE float3 bitselect(const float3 &a, const float3 &b, const int3 &c)
    {
        return float3(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y), bitselect(a.z, b.z, c.z));
    }

    static MATH_FORCEINLINE float2 bitselect(const float2 &a, const float2 &b, const int2 &c)
    {
        return float2(bitselect(a.x, b.x, c.x), bitselect(a.y, b.y, c.y));
    }

#endif

    //
    //  select: sign select
    //  return value: msb(c) ? b : a
    //

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 select(const int4 &a, const int4 &b, const int4 &c)
    {
#   if defined(__ARM_NEON)
        return vbslq_s32(vreinterpretq_u32_s32(vshrq_n_s32((int32x4_t)c, 31)), (int32x4_t)b, (int32x4_t)a);
#   elif defined(__SSE4_1__)
#        define MATH_HAS_FAST_SELECT
        return int4(_mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps((__m128i)a), _mm_castsi128_ps((__m128i)b), _mm_castsi128_ps((__m128i)c))));
#   elif defined(__SSE2__)
        __m128i d = _mm_srai_epi32((__m128i)c, 31);
        return _mm_or_si128(_mm_and_si128(d, (__m128i)b), _mm_andnot_si128(d, (__m128i)a));
#   elif defined(__SSE__)
        return int4((__m128)select(float4((__m128)a), float4((__m128)b), c));
#   else
        return int4(
            (a.x & ~(c.x >> 31)) | (b.x & (c.x >> 31)),
            (a.y & ~(c.y >> 31)) | (b.y & (c.y >> 31)),
            (a.z & ~(c.z >> 31)) | (b.z & (c.z >> 31)),
            (a.w & ~(c.w >> 31)) | (b.w & (c.w >> 31)));
#   endif
    }

#else

    static MATH_FORCEINLINE int select(int a, int b, int c)
    {
        return bitselect(a, b, c >> 31);
    }

    static MATH_FORCEINLINE int4 select(const int4 &a, const int4 &b, const int4 &c)
    {
        return int4(select(a.x, b.x, c.x), select(a.y, b.y, c.y), select(a.z, b.z, c.z), select(a.w, b.w, c.w));
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 select(const float4 &a, const float4 &b, const int4 &c)
    {
#   if defined(__ARM_NEON)
        return vbslq_f32(vreinterpretq_u32_s32(vshrq_n_s32((int32x4_t)c, 31)), (float32x4_t)b, (float32x4_t)a);
#   elif defined(__SSE4_1__)
#       define MATH_HAS_FAST_SELECT
        return _mm_blendv_ps((__m128)a, (__m128)b, _mm_castsi128_ps((__m128i)c));
#   elif defined(__SSE2__)
        __m128 d = _mm_castsi128_ps(_mm_srai_epi32((__m128i)c, 31));
        return _mm_or_ps(_mm_and_ps(d, (__m128)b), _mm_andnot_ps(d, (__m128)a));
#   elif defined(__SSE__)
        __m128 d = _mm_cmplt_ps(_mm_or_ps(_mm_and_ps(_mm_castsi128_ps((__m128i)c), cv4f(-0.f, -0.f, -0.f, -0.f)), cv4f(1.f, 1.f, 1.f, 1.f)), _mm_setzero_ps());
        return _mm_or_ps(_mm_and_ps(d, (__m128)b), _mm_andnot_ps(d, (__m128)a));
#   else
        return bitselect(a, b, c >> 31);
#   endif
    }

    static MATH_FORCEINLINE float select(float a, float b, int c)
    {
#   if defined(__SSE2__)
        return _mm_cvtss_f32((__m128)select(float4(_mm_cvtf32_ss(a)), float4(_mm_cvtf32_ss(b)), int4(_mm_cvtsi32_si128(c))));
#   else
        return bitselect(a, b, c >> 31);
#   endif
    }

    static MATH_FORCEINLINE float3 select(const float3 &a, const float3 &b, const int3 &c)
    {
        return select(float4(as_native(a)), float4(as_native(b)), int4(as_native(c))).xyz;
    }

    static MATH_FORCEINLINE float2 select(const float2 &a, const float2 &b, const int2 &c)
    {
        return select(float4(as_native(a)), float4(as_native(b)), int4(as_native(c))).xy;
    }

    static MATH_FORCEINLINE float1 select(const float1 &a, const float1 &b, const int1 &c)
    {
        return float1((float1::packed)select(float4((float1::packed)a), float4((float1::packed)b), int4((int1::packed)c)));
    }

#else

    static MATH_FORCEINLINE float select(float a, float b, int c)
    {
        return bitselect(a, b, c >> 31);
    }

    static MATH_FORCEINLINE float4 select(const float4 &a, const float4 &b, const int4 &c)
    {
        return float4(select(a.x, b.x, c.x), select(a.y, b.y, c.y), select(a.z, b.z, c.z), select(a.w, b.w, c.w));
    }

    static MATH_FORCEINLINE float3 select(const float3 &a, const float3 &b, const int3 &c)
    {
        return float3(select(a.x, b.x, c.x), select(a.y, b.y, c.y), select(a.z, b.z, c.z));
    }

    static MATH_FORCEINLINE float2 select(const float2 &a, const float2 &b, const int2 &c)
    {
        return float2(select(a.x, b.x, c.x), select(a.y, b.y, c.y));
    }

#endif

//////////////////////////////////////
    //
    //  cond: ternary operator ? :
    //  return value: c ? a : b
    //
    static MATH_FORCEINLINE float4 cond(bool c, const float4 &a, const float4 &b)
    {
        return select(b, a, -int4(c));
    }

    static MATH_FORCEINLINE float cond(bool c, float a, float b)
    {
        return select(b, a, -int(c));
    }

    static MATH_FORCEINLINE float3 cond(bool c, const float3 &a, const float3 &b)
    {
        return select(b, a, -int3(c));
    }

    static MATH_FORCEINLINE float2 cond(bool c, const float2 &a, const float2 &b)
    {
        return select(b, a, -int2(c));
    }

#if MATH_HAS_SIMD_FLOAT
    static MATH_FORCEINLINE float1 cond(bool c, const float1 &a, const float1 &b)
    {
        return select(b, a, -int1(c));
    }

#endif


    ///////////////////////////////////////////////////////////
    //
    //  sign: signum
    //  return value: -1 if x < 0, 0 if x == 0, 1 if x > 0
    //  note: the return value for NaNs is left to the implementation
    //
#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 sign(const int4 &x)
    {
#   if defined(__SSSE3__)
        return _mm_sign_epi32((__m128i)cv4i(1, 1, 1, 1), (__m128i)x);
#   else
        return (x < 0) - (x > 0);
#   endif
    }

    static MATH_FORCEINLINE int sign(int x)
    {
        return (x > 0) - (x < 0);
    }

    static MATH_FORCEINLINE int3 sign(const int3 &x)
    {
        return sign(int4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE int2 sign(const int2 &x)
    {
        return sign(int4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE int1 sign(const int1 &x)
    {
        return int1((int1::packed)sign(int4((int1::packed)x)));
    }

#else

    static MATH_FORCEINLINE int sign(int x)
    {
        return (x > 0) - (x < 0);
    }

    static MATH_FORCEINLINE int4 sign(const int4 &x)
    {
        return int4(sign(x.x), sign(x.y), sign(x.z), sign(x.w));
    }

    static MATH_FORCEINLINE int3 sign(const int3 &x)
    {
        return int3(sign(x.x), sign(x.y), sign(x.z));
    }

    static MATH_FORCEINLINE int2 sign(const int2 &x)
    {
        return int2(sign(x.x), sign(x.y));
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 sign(const float4 &x)
    {
        return convert_float4((x < 0.f) - (x > 0.f));
    }

    static MATH_FORCEINLINE float sign(float x)
    {
#   if defined(__SSE2__)
        __m128 y = _mm_cvtf32_ss(x), z = _mm_setzero_ps();
        return _mm_cvtss_f32(_mm_cvtepi32_ps(_mm_sub_epi32(_mm_castps_si128(_mm_cmplt_ss(y, z)), _mm_castps_si128(_mm_cmpgt_ss(y, z)))));
#   else
        return convert_float((x > 0.f) - (x < 0.f));
#   endif
    }

    static MATH_FORCEINLINE float3 sign(const float3 &x)
    {
        return sign(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 sign(const float2 &x)
    {
        return sign(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 sign(const float1 &x)
    {
        return float1((float1::packed)sign(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float sign(float x)
    {
        return convert_float((x > 0.f) - (x < 0.f));
    }

    static MATH_FORCEINLINE float4 sign(const float4 &x)
    {
        return float4(sign(x.x), sign(x.y), sign(x.z), sign(x.w));
    }

    static MATH_FORCEINLINE float3 sign(const float3 &x)
    {
        return float3(sign(x.x), sign(x.y), sign(x.z));
    }

    static MATH_FORCEINLINE float2 sign(const float2 &x)
    {
        return float2(sign(x.x), sign(x.y));
    }

#endif

    //
    //  floor: round to integer towards negative infinity
    //  return value: closest floating-point integer less than or equal to x
    //  note: the return value for NaNs and values out of integer range are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

#   if defined(__SSE__) && !defined(__SSE2__)
namespace detail
{
    static MATH_FORCEINLINE __m128 _mm_rint_ps(__m128 p)
    {
        p = _mm_cvtsi32_ss(p, _mm_cvtss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvtss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvtss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvtss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        return p;
    }
}
#   endif

    static MATH_FORCEINLINE float floor(float x)
    {
#   if defined(__GNUC__)
        return __builtin_floorf(x);
#   else
        return ::floorf(x);
#   endif
    }

    static MATH_FORCEINLINE float4 floor(const float4 &x)
    {
#   if defined(__ARM_NEON)
        float32x4_t c = vcvtq_f32_s32(vcvtq_s32_f32((float32x4_t)x));
        return vsubq_f32(c, vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_u32(vcgtq_f32(c, (float32x4_t)x)), vreinterpretq_s32_f32(cv4f(1.f, 1.f, 1.f, 1.f)))));
#   elif defined(__SSE4_1__)
        return _mm_round_ps((__m128)x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
#   elif defined(__SSE2__)
        __m128 c = _mm_cvtepi32_ps(_mm_cvtps_epi32((__m128)x));
        return _mm_sub_ps(c, _mm_and_ps(_mm_cmpgt_ps(c, (__m128)x), cv4f(1.f, 1.f, 1.f, 1.f)));
#   elif defined(__SSE__)
        __m128 c = detail::_mm_rint_ps((__m128)x);
        return _mm_sub_ps(c, _mm_and_ps(_mm_cmpgt_ps(c, (__m128)x), cv4f(1.f, 1.f, 1.f, 1.f)));
#   else
        return float4(floor((float)x.x), floor((float)x.y), floor((float)x.z), floor((float)x.w));
#   endif
    }

    static MATH_FORCEINLINE float3 floor(const float3 &x)
    {
        return floor(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 floor(const float2 &x)
    {
        return floor(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 floor(const float1 &x)
    {
        return float1((float1::packed)floor(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float floor(float x)
    {
        return ::floorf(x);
    }

    static MATH_FORCEINLINE float4 floor(const float4 &x)
    {
        return float4(::floorf(x.x), ::floorf(x.y), ::floorf(x.z), ::floorf(x.w));
    }

    static MATH_FORCEINLINE float3 floor(const float3 &x)
    {
        return float3(::floorf(x.x), ::floorf(x.y), ::floorf(x.z));
    }

    static MATH_FORCEINLINE float2 floor(const float2 &x)
    {
        return float2(::floorf(x.x), ::floorf(x.y));
    }

#endif

    //
    //  ceil: round to integer towards positive infinity
    //  return value: closest floating-point integer greater than or equal to x
    //  note: the return value for Nans and values out of integer range are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float ceil(float x)
    {
#   if defined(__GNUC__)
        return __builtin_ceilf(x);
#   else
        return ::ceilf(x);
#   endif
    }

    static MATH_FORCEINLINE float4 ceil(const float4 &x)
    {
#   if defined(__ARM_NEON)
        float32x4_t c = vcvtq_f32_s32(vcvtq_s32_f32((float32x4_t)x));
        return vaddq_f32(c, vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_u32(vcltq_f32(c, (float32x4_t)x)), vreinterpretq_s32_f32(cv4f(1.f, 1.f, 1.f, 1.f)))));
#   elif defined(__SSE4_1__)
        return _mm_round_ps((__m128)x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
#   elif defined(__SSE2__)
        __m128 c = _mm_cvtepi32_ps(_mm_cvtps_epi32((__m128)x));
        return _mm_add_ps(c, _mm_and_ps(_mm_cmplt_ps(c, (__m128)x), cv4f(1.f, 1.f, 1.f, 1.f)));
#   elif defined(__SSE__)
        __m128 c = detail::_mm_rint_ps((__m128)x);
        return _mm_add_ps(c, _mm_and_ps(_mm_cmplt_ps(c, (__m128)x), cv4f(1.f, 1.f, 1.f, 1.f)));
#   else
        return float4(ceil((float)x.x), ceil((float)x.y), ceil((float)x.z), ceil((float)x.w));
#   endif
    }

    static MATH_FORCEINLINE float3 ceil(const float3 &x)
    {
        return ceil(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 ceil(const float2 &x)
    {
        return ceil(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 ceil(const float1 &x)
    {
        return float1((float1::packed)ceil(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float ceil(float x)
    {
        return ::ceilf(x);
    }

    static MATH_FORCEINLINE float4 ceil(const float4 &x)
    {
        return float4(::ceilf(x.x), ::ceilf(x.y), ::ceilf(x.z), ::ceilf(x.w));
    }

    static MATH_FORCEINLINE float3 ceil(const float3 &x)
    {
        return float3(::ceilf(x.x), ::ceilf(x.y), ::ceilf(x.z));
    }

    static MATH_FORCEINLINE float2 ceil(const float2 &x)
    {
        return float2(::ceilf(x.x), ::ceilf(x.y));
    }

#endif

    //
    //  trunc: round to integer towards zero
    //  return value: closest floating-point integer who's magnitude is less than or equal to x
    //  note: the return value for NaNs and values out of integer range are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

#   if defined(__SSE__) && !defined(__SSE2__)
namespace detail
{
    static MATH_FORCEINLINE __m128 _mm_rintz_ps(__m128 p)
    {
        p = _mm_cvtsi32_ss(p, _mm_cvttss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvttss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvttss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        p = _mm_cvtsi32_ss(p, _mm_cvttss_si32(p));
        p = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 3, 2, 1));
        return p;
    }
}
#   endif

    static MATH_FORCEINLINE float4 trunc(const float4 &x)
    {
#   if defined(__ARM_NEON)
        return vcvtq_f32_s32(vcvtq_s32_f32((float32x4_t)x));
#   elif defined(__SSE4_1__)
        return _mm_round_ps((__m128)x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#   elif defined(__SSE2__)
        return _mm_cvtepi32_ps(_mm_cvttps_epi32((__m128)x));
#   elif defined(__SSE__)
        return detail::_mm_rintz_ps((__m128)x);
#   else
        return convert_float4(convert_int4(x));
#   endif
    }

    static MATH_FORCEINLINE float trunc(float x)
    {
#   if defined(_MSC_VER)
        return (float)(int)x;
#   elif defined(__GNUC__)
        return __builtin_truncf(x);
#   else
        return ::truncf(x);
#   endif
    }

    static MATH_FORCEINLINE float3 trunc(const float3 &x)
    {
        return trunc(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 trunc(const float2 &x)
    {
        return trunc(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 trunc(const float1 &x)
    {
        return float1((float1::packed)trunc(float4((float1::packed)x)));
    }

#else


    static MATH_FORCEINLINE float trunc(float x)
    {
#   if defined(_MSC_VER)
        return (float)(int)x;
#   elif defined(__ARMCC_VERSION)
        return __builtin_truncf(x);
#   else
        return ::truncf(x);
#   endif
    }

    static MATH_FORCEINLINE float4 trunc(const float4 &x)
    {
        return float4(trunc(x.x), trunc(x.y), trunc(x.z), trunc(x.w));
    }

    static MATH_FORCEINLINE float3 trunc(const float3 &x)
    {
        return float3(trunc(x.x), trunc(x.y), trunc(x.z));
    }

    static MATH_FORCEINLINE float2 trunc(const float2 &x)
    {
        return float2(trunc(x.x), trunc(x.y));
    }

#endif

    //
    //  chgsign: change sign
    //  return value: msb(y) ? -x : x
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 chgsign(const float4 &x, const float4 &y)
    {
        return as_float4(as_int4((float1::packed)x) ^ (as_int4((float1::packed)y) & (int)0x80000000));
    }

    static MATH_FORCEINLINE float chgsign(float x, float y)
    {
#   if defined(__SSE__)
        return _mm_cvtss_f32((__m128)chgsign(float4(_mm_cvtf32_ss(x)), float4(_mm_cvtf32_ss(y))));
#   else
        return as_float(as_int(x) ^ (as_int(y) & (int)0x80000000));
#   endif
    }

    static MATH_FORCEINLINE float3 chgsign(const float3 &x, const float3 &y)
    {
        return chgsign(float4(as_native(x)), float4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE float2 chgsign(const float2 &x, const float2 &y)
    {
        return chgsign(float4(as_native(x)), float4(as_native(y))).xy;
    }

    static MATH_FORCEINLINE float1 chgsign(const float1 &x, const float1 &y)
    {
        return float1((float1::packed)chgsign(float4((float1::packed)x), float4((float1::packed)y)));
    }

#else

    static MATH_FORCEINLINE float chgsign(float x, float y)
    {
        return as_float(as_int(x) ^ (as_int(y) & (int)0x80000000));
    }

    static MATH_FORCEINLINE float4 chgsign(const float4 &x, const float4 &y)
    {
        return float4(chgsign(x.x, y.x), chgsign(x.y, y.y), chgsign(x.z, y.z), chgsign(x.w, y.w));
    }

    static MATH_FORCEINLINE float3 chgsign(const float3 &x, const float3 &y)
    {
        return float3(chgsign(x.x, y.x), chgsign(x.y, y.y), chgsign(x.z, y.z));
    }

    static MATH_FORCEINLINE float2 chgsign(const float2 &x, const float2 &y)
    {
        return float2(chgsign(x.x, y.x), chgsign(x.y, y.y));
    }

#endif

    //
    //  round: round to nearest integer (ties are left to the implementation)
    //  return value: closest floating-point integer to x
    //  note: the return value for NaNs and values out of integer range are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 round(const float4 &x)
    {
#   if defined(__ARM_NEON)
        float32x4_t c = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(cv4f(8388608., 8388608., 8388608., 8388608.)), vandq_s32(vreinterpretq_s32_f32((float32x4_t)x), cv4i(0x80000000, 0x80000000, 0x80000000, 0x80000000))));
        return vsubq_f32(vaddq_f32((float32x4_t)x, c), c);
#   elif defined(__SSE4_1__)
        return _mm_round_ps((__m128)x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#   elif defined(__SSE2__)
        return _mm_cvtepi32_ps(_mm_cvtps_epi32((__m128)x));
#   elif defined(__SSE__)
        return detail::_mm_rint_ps((__m128)x);
#   else
        float4 c = chgsign(float4(8388608.), x);
        return (x + c) - c;
#   endif
    }

    static MATH_FORCEINLINE float round(float x)
    {
#   if defined(_MSC_VER)
        float y = x, c = 8388608.f;
        return y >= 0.f ? float(float(y + c) - c) : float(float(y - c) + c);
#   elif defined(__GNUC__)
        return __builtin_roundf(x);
#   else
        return ::roundf(x);
#   endif
    }

    static MATH_FORCEINLINE float3 round(const float3 &x)
    {
        return round(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 round(const float2 &x)
    {
        return round(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 round(const float1 &x)
    {
        return float1((float1::packed)round(float4((float1::packed)x)));
    }

#else


    static MATH_FORCEINLINE float round(float x)
    {
#   if defined(_MSC_VER)
        float y = x, c = 8388608.f;
        return y >= 0.f ? float(float(y + c) - c) : float(float(y - c) + c);
#   elif defined(__ARMCC_VERSION)
        return __builtin_roundf(x);
#   else
        return ::roundf(x);
#   endif
    }

    static MATH_FORCEINLINE float4 round(const float4 &x)
    {
        return float4(round(x.x), round(x.y), round(x.z), round(x.w));
    }

    static MATH_FORCEINLINE float3 round(const float3 &x)
    {
        return float3(round(x.x), round(x.y), round(x.z));
    }

    static MATH_FORCEINLINE float2 round(const float2 &x)
    {
        return float2(round(x.x), round(x.y));
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 copysign(const float4 &x, const float4 &y)
    {
        return chgsign(abs(x), y);
    }

    static MATH_FORCEINLINE float copysign(float x, float y)
    {
#   if defined(__GNUC__)
        return __builtin_copysignf(x, y);
#   else
        return chgsign(abs(x), y);
#   endif
    }

    static MATH_FORCEINLINE float3 copysign(const float3 &x, const float3 &y)
    {
        return chgsign(abs(x), y);
    }

    static MATH_FORCEINLINE float2 copysign(const float2 &x, const float2 &y)
    {
        return chgsign(abs(x), y);
    }

    static MATH_FORCEINLINE float1 copysign(const float1 &x, const float1 &y)
    {
        return chgsign(abs(x), y);
    }

#else

    static MATH_FORCEINLINE float copysign(float x, float y)
    {
#   if defined(_MSC_VER)
        return (float)::_copysign((double)x, (double)y);
#   else
        return ::copysignf(x, y);
#   endif
    }

    static MATH_FORCEINLINE float4 copysign(const float4 &x, const float4 &y)
    {
        return float4(copysign(x.x, y.x), copysign(x.y, y.y), copysign(x.z, y.z), copysign(x.w, y.w));
    }

    static MATH_FORCEINLINE float3 copysign(const float3 &x, const float3 &y)
    {
        return float3(copysign(x.x, y.x), copysign(x.y, y.y), copysign(x.z, y.z));
    }

    static MATH_FORCEINLINE float2 copysign(const float2 &x, const float2 &y)
    {
        return float2(copysign(x.x, y.x), copysign(x.y, y.y));
    }

#endif

    //
    //  isfinite
    //  returns 0 if the number is non-finite (nan, inf, -inf)
    //  float2/3/4 versions return ~0 when the number is finite
    //  float/float1 version return 1 when the number is finite
#   if defined(isfinite)
#       undef isfinite
#   endif

    static MATH_FORCEINLINE int4 isfinite(const float4 &x)
    {
        return ((as_int4(x) & 0x7f800000) != 0x7f800000);
    }

    static MATH_FORCEINLINE int3 isfinite(const float3 &x)
    {
        return ((as_int3(x) & 0x7f800000) != 0x7f800000);
    }

    static MATH_FORCEINLINE int2 isfinite(const float2 &x)
    {
        return ((as_int2(x) & 0x7f800000) != 0x7f800000);
    }

    static MATH_FORCEINLINE int isfinite(float x)
    {
        return ((as_int(x) & 0x7f800000) != 0x7f800000);
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE int1 isfinite(const float1& x)
    {
#   if defined(MATH_HAS_SIMD_INT)
        return -int1((int1::packed)isfinite(float4((float1::packed)x)));
#   else
        return -isfinite(float4((float1::packed)x)).x;
#   endif
    }

#endif

    //
    //  rcpe: reciprocal estimate
    //  return value: fast approximation to 1/x, with rcpe(1) := 1
    //  note: the return value for denormals, NaNs, zero and infinity are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 rcpe(const float4 &x)
    {
#   if defined(__ARM_NEON)
#       define C0  1.0019569e+00f
        return vmulq_f32(vrecpeq_f32((float32x4_t)x), cv4f(C0, C0, C0, C0));
#       undef C0
#   elif defined(__SSE__)
    #   define C0  1.0002443e+00f
        return _mm_mul_ps(_mm_rcp_ps((__m128)x), cv4f(C0, C0, C0, C0));
    #   undef C0
#   else
        int4 i = as_int4(abs(x));
        float4 x0 = chgsign(as_float4(0x7ef311c1 - i), x);
        return x0 * (2.002687e+00f - x * x0);
#   endif
    }

    static MATH_FORCEINLINE float rcpe(float x)
    {
#   if defined(__SSE__)
    #   define C0  1.0002443e+00f
        return _mm_cvtss_f32(_mm_mul_ss(_mm_rcp_ss(_mm_cvtf32_ss(x)), cv4f(C0, C0, C0, C0)));
    #   undef C0
#   else
        int i = as_int(abs(x));
        float x0 = chgsign(as_float(0x7ef311c1 - i), x);
        return x0 * (2.002687e+00f - x * x0);
#   endif
    }

    static MATH_FORCEINLINE float3 rcpe(const float3 &x)
    {
        return rcpe(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 rcpe(const float2 &x)
    {
        return rcpe(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 rcpe(const float1 &x)
    {
        return float1((float1::packed)rcpe(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float rcpe(float x)
    {
        return 1.f / x;
    }

    static MATH_FORCEINLINE float4 rcpe(const float4 &x)
    {
        return float4(1.f / x.x, 1.f / x.y, 1.f / x.z, 1.f / x.w);
    }

    static MATH_FORCEINLINE float3 rcpe(const float3 &x)
    {
        return float3(1.f / x.x, 1.f / x.y, 1.f / x.z);
    }

    static MATH_FORCEINLINE float2 rcpe(const float2 &x)
    {
        return float2(1.f / x.x, 1.f / x.y);
    }

#endif

    //
    //  rcp: reciprocal
    //  return value: 1/x
    //  note: the return value for NaNs are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 rcp(const float4 &x)
    {
#   if defined(__ARM_NEON)
        float32x4_t e0 = vrecpeq_f32((float32x4_t)x);
        float32x4_t r = vrecpsq_f32((float32x4_t)x, e0), e = vmulq_f32(e0, r);
        r = vrecpsq_f32((float32x4_t)x, e); e = vmulq_f32(e, r);
        return vbslq_f32(vceqq_f32((float32x4_t)x, cv4f(0.f, 0.f, 0.f, 0.f)), e0, e);
# elif defined(__SSE__)
#  define C1  2.000000477f
#  define C2  2.f
        __m128 e0 = _mm_rcp_ps((__m128)x);
        __m128 e = _mm_mul_ps(e0, _mm_sub_ps(cv4f(C1, C1, C1, C1), _mm_mul_ps((__m128)x, e0)));
        e = _mm_mul_ps(e, _mm_sub_ps(cv4f(C2, C2, C2, C2), _mm_mul_ps((__m128)x, e)));
        // _mm_rcp_ps handle exceptions, but adding 2 pass of NR can yield a QNAN,
        // comparing e with e may look silly but this is how we detect that the NR pass did yield a nan
        // by definition nan != nan so in this case we take te result from _mm_rcp_ps
        return select(float4(e0), float4(e), float4(e) == float4(e));
#  undef C2
#  undef C1
# else
        return 1.f / x;
#   endif
    }

    static MATH_FORCEINLINE float rcp(float x)
    {
        return 1.f / x;
    }

    static MATH_FORCEINLINE float3 rcp(const float3 &x)
    {
        return rcp(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 rcp(const float2 &x)
    {
        return rcp(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 rcp(const float1 &x)
    {
        return float1((float1::packed)rcp(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float rcp(float x)
    {
        return 1.f / x;
    }

    static MATH_FORCEINLINE float4 rcp(const float4 &x)
    {
        return float4(1.f / x.x, 1.f / x.y, 1.f / x.z, 1.f / x.w);
    }

    static MATH_FORCEINLINE float3 rcp(const float3 &x)
    {
        return float3(1.f / x.x, 1.f / x.y, 1.f / x.z);
    }

    static MATH_FORCEINLINE float2 rcp(const float2 &x)
    {
        return float2(1.f / x.x, 1.f / x.y);
    }

#endif

    //
    //  rsqrte: square root reciprocal estimate
    //  return value: fast approximation of 1/sqrt(x), with rsqrte(1) := 1
    //  note: the return value for denormals, NaNs, negative values, zero and infinity are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    extern float4 rsqrte_magicnumber;

    static MATH_FORCEINLINE float4 rsqrte(const float4 &x)
    {
#   if defined(__ARM_NEON)
        return vmulq_f32(vrsqrteq_f32((float32x4_t)x), cv4f(1.0019569e+00f, 1.0019569e+00f, 1.0019569e+00f, 1.0019569e+00f));
#   elif defined(__SSE__)
    #   if PLATFORM_OSX
        #   define C0  1.0002443e+00f
        return _mm_mul_ps(_mm_rsqrt_ps((__m128)x), cv4f(C0, C0, C0, C0));
        #   undef C0
    #   elif PLATFORM_PS4 || PLATFORM_XBOXONE
        #   define C0  1.0001221e+00f
        return _mm_mul_ps(_mm_rsqrt_ps((__m128)x), cv4f(C0, C0, C0, C0));
        #   undef C0
    #   else // for windows or linux we don't know at compile time if the cpu will be Intel or AMD
        return _mm_mul_ps(_mm_rsqrt_ps((__m128)x), (__m128)rsqrte_magicnumber);
    #   endif
#   else
        int4 i = as_int4(x); i = 0x5f37642f - (i >> 1);
        return as_float4(i);
#   endif
    }

    static MATH_FORCEINLINE float rsqrte(float x)
    {
#   if defined(__SSE__)
    #   if PLATFORM_OSX
        #   define C0  1.0002443e+00f
        return _mm_cvtss_f32(_mm_mul_ss(_mm_rsqrt_ps(_mm_cvtf32_ss(x)), cv4f(C0, C0, C0, C0)));
        #   undef C0
    #   elif PLATFORM_PS4 || PLATFORM_XBOXONE
        #   define C0  1.0001221e+00f
        return _mm_cvtss_f32(_mm_mul_ss(_mm_rsqrt_ps(_mm_cvtf32_ss(x)), cv4f(C0, C0, C0, C0)));
        #   undef C0
    #   else // for windows or linux we don't know at compile time if the cpu will be Intel or AMD
        return _mm_cvtss_f32(_mm_mul_ss(_mm_rsqrt_ps(_mm_cvtf32_ss(x)), (__m128)rsqrte_magicnumber));
    #   endif
#   else
        // see https://en.wikipedia.org/wiki/Fast_inverse_square_root
    #   define C0  0x5f375a86
        const float threehalfs = 1.5F;

        float x2 = x * 0.5F;
        float y  = x;
        int i  = as_int(y);                         // evil floating point bit level hacking
        i  = C0 - (i >> 1);                         // what?
        y  = as_float(i);
        y  = y * (threehalfs - (x2 * y * y));       // 1st iteration Newton's method
        y  = y * (threehalfs - (x2 * y * y));       // 2nd iteration,
        return y;
    #   undef C0
#   endif
    }

    static MATH_FORCEINLINE float3 rsqrte(const float3 &x)
    {
        return rsqrte(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 rsqrte(const float2 &x)
    {
        return rsqrte(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 rsqrte(const float1 &x)
    {
        return float1((float1::packed)rsqrte(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float rsqrte(float x)
    {
        return 1.f / ::sqrtf(x);
    }

    static MATH_FORCEINLINE float4 rsqrte(const float4 &x)
    {
        return float4(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y), 1.f / ::sqrtf(x.z), 1.f / ::sqrtf(x.w));
    }

    static MATH_FORCEINLINE float3 rsqrte(const float3 &x)
    {
        return float3(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y), 1.f / ::sqrtf(x.z));
    }

    static MATH_FORCEINLINE float2 rsqrte(const float2 &x)
    {
        return float2(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y));
    }

#endif

    //
    //  rsqrt: square root reciprocal
    //  return value: 1/sqrt(x)
    //  note: the return value for NaNs and negative values are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float rsqrt(float x)
    {
#   if defined(__SSE__)
#       define C0  9.999998e-01f
#       define C1  3.0000002e+00f
#       define C2  .5f
#       define C3  340282346638528859811704183484516925440.f
        __m128 y = _mm_cvtf32_ss(x);
        __m128 e = _mm_mul_ss(_mm_rsqrt_ss(y), cv4f(C0, C0, C0, C0));
        e = _mm_min_ss(e, cv4f(C3, C3, C3, C3));
        return _mm_cvtss_f32(_mm_mul_ss(_mm_mul_ss(e, cv4f(C2, C2, C2, C2)), _mm_sub_ss(cv4f(C1, C1, C1, C1), _mm_mul_ss(_mm_mul_ss(y, e), e))));
#       undef C3
#       undef C2
#       undef C1
#       undef C0
#   elif defined(__GNUC__)
        return 1.f / __builtin_sqrtf(x);
#   else
        return 1.f / sqrtf(x);
#   endif
    }

    static MATH_FORCEINLINE float4 rsqrt(const float4 &x)
    {
#   if defined(__ARM_NEON)
        float32x4_t e0 = vrsqrteq_f32((float32x4_t)x), r = vrsqrtsq_f32(vmulq_f32(e0, (float32x4_t)x), e0);
        float32x4_t e = vmulq_f32(e0, r); r = vrsqrtsq_f32(vmulq_f32(e, (float32x4_t)x), e);
        return vbslq_f32(vceqq_f32((float32x4_t)x, cv4f(0.f, 0.f, 0.f, 0.f)), e0, vmulq_f32(e, r));
#   elif defined(__SSE__)
#       define C0  9.999998e-01f
#       define C1  3.0000002e+00f
#       define C2  .5f
#       define C3  340282346638528859811704183484516925440.f
        __m128 e = _mm_mul_ps(_mm_rsqrt_ps((__m128)x), cv4f(C0, C0, C0, C0));
        e = _mm_min_ps(e, cv4f(C3, C3, C3, C3));
        return _mm_mul_ps(_mm_mul_ps(e, cv4f(C2, C2, C2, C2)), _mm_sub_ps(cv4f(C1, C1, C1, C1), _mm_mul_ps(_mm_mul_ps((__m128)x, e), e)));
#       undef C3
#       undef C2
#       undef C1
#       undef C0
#   else
        return float4(rsqrt(x.x), rsqrt(x.y), rsqrt(x.z), rsqrt(x.w));
#   endif
    }

    static MATH_FORCEINLINE float3 rsqrt(const float3 &x)
    {
        return rsqrt(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 rsqrt(const float2 &x)
    {
        return rsqrt(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 rsqrt(const float1 &x)
    {
        return float1((float1::packed)rsqrt(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float rsqrt(float x)
    {
        return 1.f / ::sqrtf(x);
    }

    static MATH_FORCEINLINE float4 rsqrt(const float4 &x)
    {
        return float4(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y), 1.f / ::sqrtf(x.z), 1.f / ::sqrtf(x.w));
    }

    static MATH_FORCEINLINE float3 rsqrt(const float3 &x)
    {
        return float3(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y), 1.f / ::sqrtf(x.z));
    }

    static MATH_FORCEINLINE float2 rsqrt(const float2 &x)
    {
        return float2(1.f / ::sqrtf(x.x), 1.f / ::sqrtf(x.y));
    }

#endif

    //
    //  sqrt: square root
    //  return value: sqrt(x)
    //  note: the return value for NaNs and negative values are left to the implementation
    //
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 sqrt(const float4 &x)
    {
#   if defined(__aarch64__)
        return vsqrtq_f32(x);
#   elif defined(_M_ARM64)
        // On Windows arm64, vsqrtq_f32 is defined via #define. When built lumped, sometimes the define will not be available. The intrinsic in MSVC is instead neon_fsqrt32.
        return neon_fsqrtq32((float32x4_t)x);
#   elif defined(__ARM_NEON)
        float32x4_t e0 = vrsqrteq_f32((float32x4_t)x), r = vrsqrtsq_f32(vmulq_f32(e0, (float32x4_t)x), e0);
        float32x4_t e = vmulq_f32(e0, r); r = vrsqrtsq_f32(vmulq_f32(e, (float32x4_t)x), e);
        return vbslq_f32(vceqq_f32((float32x4_t)x, cv4f(0.f, 0.f, 0.f, 0.f)), (float32x4_t)x, vmulq_f32((float32x4_t)x, vmulq_f32(e, r)));
#   elif defined(__SSE__)
#       define MATH_HAS_FAST_SQRT
        return _mm_sqrt_ps((__m128)x);
#   elif defined(MATH_HAS_FAST_SELECT)
        return select(x * rsqrt(x), x, x == float4(ZERO));
#   else
        return bitselect(x * rsqrt(x), x, x == float4(ZERO));
#   endif
    }

    static MATH_FORCEINLINE float sqrt(float x)
    {
#   if defined(__GNUC__)
        return __builtin_sqrtf(x);
#   else
        return sqrtf(x);
#   endif
    }

    static MATH_FORCEINLINE float3 sqrt(const float3 &x)
    {
        return sqrt(float4(as_native(x))).xyz;
    }

    static MATH_FORCEINLINE float2 sqrt(const float2 &x)
    {
        return sqrt(float4(as_native(x))).xy;
    }

    static MATH_FORCEINLINE float1 sqrt(const float1 &x)
    {
        return float1((float1::packed)sqrt(float4((float1::packed)x)));
    }

#else

    static MATH_FORCEINLINE float sqrt(float x)
    {
        return ::sqrtf(x);
    }

    static MATH_FORCEINLINE float4 sqrt(const float4 &x)
    {
        return float4(::sqrtf(x.x), ::sqrtf(x.y), ::sqrtf(x.z), ::sqrtf(x.w));
    }

    static MATH_FORCEINLINE float3 sqrt(const float3 &x)
    {
        return float3(::sqrtf(x.x), ::sqrtf(x.y), ::sqrtf(x.z));
    }

    static MATH_FORCEINLINE float2 sqrt(const float2 &x)
    {
        return float2(::sqrtf(x.x), ::sqrtf(x.y));
    }

#endif

    static MATH_FORCEINLINE float saturate(float v)
    {
        return min(1.f, max(0.f, v));
    }

    static MATH_FORCEINLINE float4 saturate(const float4 & v)
    {
        return min(float4(1.f), max(float4(ZERO), v));
    }

    static MATH_FORCEINLINE float3 saturate(const float3 & v)
    {
        return min(float3(1.f), max(float3(ZERO), v));
    }

    static MATH_FORCEINLINE float2 saturate(const float2 & v)
    {
        return min(float2(1.f), max(float2(ZERO), v));
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float1 saturate(const float1& v)
    {
        return min(float1(1.f), max(float1(ZERO), v));
    }

#endif

    static MATH_FORCEINLINE float clamp(float x, float a, float b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE int clamp(int x, int a, int b)
    {
        return min(max(x, a), b);
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 clamp(const float4 &x, const float4 &a, const float4 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE float3 clamp(const float3 &x, const float3 &a, const float3 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE float2 clamp(const float2 &x, const float2 &a, const float2 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE float1 clamp(const float1 &x, const float1 &a, const float1 &b)
    {
        return min(max(x, a), b);
    }

#else

    static MATH_FORCEINLINE float4 clamp(const float4 &x, const float4 &a, const float4 &b)
    {
        return float4(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z), clamp(x.w, a.w, b.w));
    }

    static MATH_FORCEINLINE float3 clamp(const float3 &x, const float3 &a, const float3 &b)
    {
        return float3(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z));
    }

    static MATH_FORCEINLINE float2 clamp(const float2 &x, const float2 &a, const float2 &b)
    {
        return float2(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y));
    }

#endif

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 clamp(const int4 &x, const int4 &a, const int4 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE int3 clamp(const int3 &x, const int3 &a, const int3 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE int2 clamp(const int2 &x, const int2 &a, const int2 &b)
    {
        return min(max(x, a), b);
    }

    static MATH_FORCEINLINE int1 clamp(const int1 &x, const int1 &a, const int1 &b)
    {
        return min(max(x, a), b);
    }

#else

    static MATH_FORCEINLINE int4 clamp(const int4 &x, const int4 &a, const int4 &b)
    {
        return int4(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z), clamp(x.w, a.w, b.w));
    }

    static MATH_FORCEINLINE int3 clamp(const int3 &x, const int3 &a, const int3 &b)
    {
        return int3(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y), clamp(x.z, a.z, b.z));
    }

    static MATH_FORCEINLINE int2 clamp(const int2 &x, const int2 &a, const int2 &b)
    {
        return int2(clamp(x.x, a.x, b.x), clamp(x.y, a.y, b.y));
    }

#endif

    //
    //  csum: vector sum
    //  return value: the horizontal sum of a vector
    //  note: the return value for NaNs are left to the implementation
    //
    static MATH_FORCEINLINE float1 csum(const float4 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return (p.x + p.y) + (p.z + p.w);
#   elif defined(__ARM_NEON)
        float32x2_t r = vpadd_f32(vget_low_f32((float32x4_t)p), vget_high_f32((float32x4_t)p));
        return float1(vpadd_f32(r, r));
#   elif defined(__SSE3__)
        __m128 r = _mm_hadd_ps((__m128)p, (__m128)p);
        return float1(_mm_hadd_ps(r, r));
#   elif defined(__SSE2__)
        __m128 r = _mm_add_ps((__m128)p, _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128((__m128)p), _MM_SHUFFLE(0, 3, 2, 1))));
        return float1(_mm_add_ps(r, _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(r), _MM_SHUFFLE(1, 0, 3, 2)))));
#   elif defined(__SSE__)
        __m128 r = _mm_add_ps((__m128)p, _mm_shuffle_ps((__m128)p, (__m128)p, _MM_SHUFFLE(0, 3, 2, 1)));
        return float1(_mm_add_ps(r, _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 0, 3, 2))));
#   else
        float4 q = p + p.yzwx; q = q + q.zwxy;
        return float1((float1::packed)q);
#   endif
    }

    static MATH_FORCEINLINE float1 csum(const float3 &p)
    {
        return csum(float4(p.xyz, 0.f));
    }

    static MATH_FORCEINLINE float1 csum(const float2 &p)
    {
        return csum(float4(p, float2(ZERO)));
    }

    //
    //  cmax: vector maximum
    //  return value: the horizontal maximum of a vector
    //  note: the return value for NaNs are left to the implementation
    //
    static MATH_FORCEINLINE int1 cmax(const int4 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return max(max(p.x, p.y), max(p.z, p.w));
#   elif defined(__ARM_NEON)
        int32x2_t r = vpmax_s32(vget_low_s32((int32x4_t)p), vget_high_s32((int32x4_t)p));
        return int1(vpmax_s32(r, r));
#   else
        int4 q = max(p, p.yzwx); q = max(q, q.zwxy);
        return int1((int1::packed)q);
#   endif
    }

    static MATH_FORCEINLINE int1 cmax(const int3 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return max(max(p.x, p.y), p.z);
#   else
        return cmax(p.xyzx);
#   endif
    }

    static MATH_FORCEINLINE int1 cmax(const int2 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return max(p.x, p.y);
#   elif defined(__ARM_NEON) && defined(MATH_HAS_NATIVE_SIMD)
        int32x2_t r = (int32x2_t)p;
        return int1(vpmax_s32(r, r));
#   elif defined(__ARM_NEON)
        int32x2_t r = vget_low_s32((int32x4_t)p);
        return int1(vpmax_s32(r, r));
#   else
        return cmax(p.xxyy);
#   endif
    }

    static MATH_FORCEINLINE float1 cmax(const float4 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return max(max(p.x, p.y), max(p.z, p.w));
#   elif defined(__ARM_NEON)
        float32x2_t r = vpmax_f32(vget_low_f32((float32x4_t)p), vget_high_f32((float32x4_t)p));
        return float1(vpmax_f32(r, r));
#   else
        float4 q = max(p, p.yzwx); q = max(q, q.zwxy);
        return float1((float1::packed)q);
#   endif
    }

    static MATH_FORCEINLINE float1 cmax(const float3 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return max(max(p.x, p.y), p.z);
#   else
        return cmax(p.xyzx);
#   endif
    }

    static MATH_FORCEINLINE float1 cmax(const float2 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return max(p.x, p.y);
#   else
        return cmax(p.xxyy);
#   endif
    }

    //
    //  cmin: vector minimum
    //  return value: the horizontal minimum of a vector
    //  note: the return value for NaNs are left to the implementation
    //
    static MATH_FORCEINLINE int1 cmin(const int4 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return min(min(p.x, p.y), min(p.z, p.w));
#   elif defined(__ARM_NEON)
        int32x2_t r = vpmin_s32(vget_low_s32((int32x4_t)p), vget_high_s32((int32x4_t)p));
        return int1(vpmin_s32(r, r));
#   else
        int4 q = min(p, p.yzwx); q = min(q, q.zwxy);
        return int1((int1::packed)q);
#   endif
    }

    static MATH_FORCEINLINE int1 cmin(const int3 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return min(min(p.x, p.y), p.z);
#   else
        return cmin(p.xyzx);
#   endif
    }

    static MATH_FORCEINLINE int1 cmin(const int2 &p)
    {
#   if !defined(MATH_HAS_SIMD_INT)
        return min(p.x, p.y);
#   elif defined(__ARM_NEON) && defined(MATH_HAS_NATIVE_SIMD)
        int32x2_t r = (int32x2_t)p;
        return int1(vpmin_s32(r, r));
#   elif defined(__ARM_NEON)
        int32x2_t r = vget_low_s32((int32x4_t)p);
        return int1(vpmin_s32(r, r));
#   else
        return cmin(p.xxyy);
#   endif
    }

    static MATH_FORCEINLINE float1 cmin(const float4 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return min(min(p.x, p.y), min(p.z, p.w));
#   elif defined(__ARM_NEON)
        float32x2_t r = vpmin_f32(vget_low_f32((float32x4_t)p), vget_high_f32((float32x4_t)p));
        return float1(vpmin_f32(r, r));
#   else
        float4 q = min(p, p.yzwx); q = min(q, q.zwxy);
        return float1((float1::packed)q);
#   endif
    }

    static MATH_FORCEINLINE float1 cmin(const float3 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return min(min(p.x, p.y), p.z);
#   else
        return cmin(p.xyzx);
#   endif
    }

    static MATH_FORCEINLINE float1 cmin(const float2 &p)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return min(p.x, p.y);
#   else
        return cmin(p.xxyy);
#   endif
    }

    //
    //  dot: dot product
    //  return value: the dot product of two vectors
    //  note: the return value for NaNs are left to the implementation
    //
    static MATH_FORCEINLINE float1 dot(const float4 &p0, const float4 &p1)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return csum(p0 * p1);
#   elif defined(__SSE4_1__)
        return float1(_mm_dp_ps((__m128)p0, (__m128)p1, 0xff));
#   else
        return csum(p0 * p1);
#   endif
    }

    static MATH_FORCEINLINE float1 dot(const float3 &p0, const float3 &p1)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return csum(p0 * p1);
#   elif defined(__SSE4_1__)
        return float1(_mm_dp_ps(as_native(p0), as_native(p1), 0x7f));
#   else
        return csum(p0 * p1);
#   endif
    }

    static MATH_FORCEINLINE float1 dot(const float2 &p0, const float2 &p1)
    {
#   if !defined(MATH_HAS_SIMD_FLOAT)
        return csum(p0 * p1);
#   elif defined(__SSE4_1__)
        return float1(_mm_dp_ps(as_native(p0), as_native(p1), 0x3f));
#   else
        return csum(p0 * p1);
#   endif
    }

    static MATH_FORCEINLINE float1 dot(const float4 &p)
    {
        return dot(p, p);
    }

    static MATH_FORCEINLINE float1 dot(const float3 &p)
    {
        return dot(p, p);
    }

    static MATH_FORCEINLINE float1 dot(const float2 &p)
    {
        return dot(p, p);
    }

    static MATH_FORCEINLINE float1 length(const float4 &v)
    {
        return sqrt(dot(v));
    }

    static MATH_FORCEINLINE float1 length(const float3 &v)
    {
        return sqrt(dot(v));
    }

    static MATH_FORCEINLINE float1 length(const float2 &v)
    {
        return sqrt(dot(v));
    }

    static MATH_FORCEINLINE float4 normalize(const float4 &v)
    {
        return v * rsqrt(dot(v));
    }

    static MATH_FORCEINLINE float3 normalize(const float3 &v)
    {
        return v * rsqrt(dot(v));
    }

    static MATH_FORCEINLINE float2 normalize(const float2 &v)
    {
        return v * rsqrt(dot(v));
    }

    static MATH_FORCEINLINE float4 normalizeSafe(const float4& v, const float4& def = float4(ZERO), const float epsilon = epsilon_normal())
    {
        float1 len = dot(v);
        return select(def, v * rsqrt(len), len > float4(epsilon));
    }

    static MATH_FORCEINLINE float3 normalizeSafe(const float3& v, const float3& def = float3(ZERO), const float epsilon = epsilon_normal())
    {
        float1 len = dot(v);
        return select(def, v * rsqrt(len), len > float3(epsilon));
    }

    static MATH_FORCEINLINE float2 normalizeSafe(const float2& v, const float2& def = float2(ZERO), const float epsilon = epsilon_normal())
    {
        float1 len = dot(v);
        return select(def, v * rsqrt(len), len > float2(epsilon));
    }

    static MATH_FORCEINLINE math::int4 normalizeToByte(const math::float4& value)
    {
        return math::convert_int4(math::saturate(value) * 255.0f + 0.5f);
    }

    //
    //  cross: cross product
    //  return value: the cross product of two vectors, zeroing the resulting 'w' compoment for 4-vectors
    //  note: the return value for NaNs are left to the implementation
    //
    static MATH_FORCEINLINE float3 cross(const float3 &p0, const float3 &p1)
    {
        return (p0 * p1.yzx - p0.yzx * p1).yzx;
    }

    static MATH_FORCEINLINE float4 cross(const float4 &p0, const float4 &p1)
    {
        return (p0 * p1.yzxw - p0.yzxw * p1).yzxw;
    }

    static MATH_FORCEINLINE float4 poly2(const float4 &x, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        float4 x2 = x * x;
        return mad(x2, c2, mad(x, c1, c0));
    }

    static MATH_FORCEINLINE float4 poly3(const float4 &x, const float4 &c3, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        float4 x2 = x * x;
        return mad(x2, mad(x, c3, c2), mad(x, c1, c0));
    }

    static MATH_FORCEINLINE float4 poly4(const float4 &x, const float4 &c4, const float4 &c3, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        return poly2(x * x, c4, mad(x, c3, c2), mad(x, c1, c0));
    }

    static MATH_FORCEINLINE float4 poly5(const float4 &x, const float4 &c5, const float4 &c4, const float4 &c3, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        return poly2(x * x, mad(x, c5, c4), mad(x, c3, c2), mad(x, c1, c0));
    }

    static MATH_FORCEINLINE float4 poly6(const float4 &x, const float4 &c6, const float4 &c5, const float4 &c4, const float4 &c3, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        return mad(x, mad(x, mad(x, mad(x, mad(x, mad(x, c6, c5), c4), c3), c2), c1), c0);
    }

    static MATH_FORCEINLINE float4 poly7(const float4 &x, const float4 &c7, const float4 &c6, const float4 &c5, const float4 &c4, const float4 &c3, const float4 &c2, const float4 &c1, const float4 &c0)
    {
        return mad(x, mad(x, mad(x, mad(x, mad(x, mad(x, mad(x, c7, c6), c5), c4), c3), c2), c1), c0);
    }

    static MATH_FORCEINLINE float degrees(float rad)
    {
        return M_RAD_2_DEG * rad;
    }

    static MATH_FORCEINLINE float4 degrees(const float4& rad)
    {
        return float1(M_RAD_2_DEG) * rad;
    }

    static MATH_FORCEINLINE float3 degrees(const float3& rad)
    {
        return float1(M_RAD_2_DEG) * rad;
    }

    static MATH_FORCEINLINE float2 degrees(const float2& rad)
    {
        return float1(M_RAD_2_DEG) * rad;
    }

#if defined(MATH_HAS_SIMD_FLOAT)
    static MATH_FORCEINLINE float1 degrees(const float1& rad)
    {
        return float1(M_RAD_2_DEG) * rad;
    }

#endif

    static MATH_FORCEINLINE float radians(float deg)
    {
        return M_DEG_2_RAD * deg;
    }

    static MATH_FORCEINLINE float4 radians(const float4& deg)
    {
        return float1(M_DEG_2_RAD) * deg;
    }

    static MATH_FORCEINLINE float3 radians(const float3& deg)
    {
        return float1(M_DEG_2_RAD) * deg;
    }

    static MATH_FORCEINLINE float2 radians(const float2& deg)
    {
        return float1(M_DEG_2_RAD) * deg;
    }

#if defined(MATH_HAS_SIMD_FLOAT)
    static MATH_FORCEINLINE float1 radians(const float1& deg)
    {
        return float1(M_DEG_2_RAD) * deg;
    }

#endif

    static MATH_FORCEINLINE float4 log2e(const float4 &x)
    {
        const int4 c = int4(0x3f800000);
        int4 i = as_int4(x);
        float4 y = as_float4((i & (int)0x807fffff) | c) - 1.f;
        float4 e = convert_float4(i >> 23) - 127.f;
        return poly3(y, 0.194381127444802140477238f, -0.632288225018497101870229f, 1.43790709757370733020881f, e);
    }

    static MATH_FORCEINLINE float4 exp2e(const float4 &x)
    {
        float4 c = max(x, -127.f), b = floor(c);
        return poly2(c - b, 0.317729908411189240473336f, 0.682270091588810729452277f, 1.000000000f) * as_float4((convert_int4(b) + 127) << 23);
    }

    static MATH_FORCEINLINE float4 powr(const float4 &x, const float4 &y)
    {
        return exp2e(y * log2e(x));
    }

    static MATH_FORCEINLINE float3 powr(const float3 &x, const float3 &y)
    {
        return powr(float4(as_native(x)), float4(as_native(y))).xyz;
    }

    static MATH_FORCEINLINE float2 powr(const float2 &x, const float2 &y)
    {
        return powr(float4(as_native(x)), float4(as_native(y))).xy;
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float1 powr(const float1 &x, const float1 &y)
    {
        return float1((float1::packed)powr(float4((float1::packed)x),  float4((float1::packed)y)));
    }

#endif

#if defined(_MSC_VER)

    static MATH_FORCEINLINE float powr(float x, float y)
    {
        return exp(y * log(x));
    }

#elif defined(__ARMCC_VERSION)

    static MATH_FORCEINLINE float powr(float x, float y)
    {
        return __builtin_exp2f(y * __builtin_log2f(x));
    }

#else

    static MATH_FORCEINLINE float powr(float x, float y)
    {
        return ::exp2f(y * ::log2f(x));
    }

#endif

    static MATH_FORCEINLINE float4 pow(const float4 &x, const float4 &y)
    {
        return float4(std::pow(x.x, y.x), std::pow(x.y, y.y), std::pow(x.z, y.z), std::pow(x.w, y.w));
    }

    static MATH_FORCEINLINE float modf(float x, float &ip)
    {
    #if defined(__GNUC__) && defined(__OPTIMIZE__) && !defined(__ghs__)
        return __builtin_modff(x, &ip);
    #else
        return std::modf(x, &ip);
    #endif
    }

    static MATH_FORCEINLINE float fmod(float x, float y)
    {
        return x - y * trunc(x / y);
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float1 fmod(const float1 &x, const float1 &y)
    {
        return x - y * trunc(float1(x / y));
    }

#endif

    static MATH_FORCEINLINE float4 fmod(const float4& x, const float4& y)
    {
        return x - y  * trunc(x / y);
    }

    static MATH_FORCEINLINE float3 fmod(const float3& x, const float3& y)
    {
        return x - y  * trunc(x / y);
    }

    static MATH_FORCEINLINE float2 fmod(const float2& x, const float2& y)
    {
        return x - y  * trunc(x / y);
    }

    static MATH_FORCEINLINE float lerp(float a, float b, float x)
    {
        return x * (b - a) + a;
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float1 lerp(const float1 &a, const float1 &b, const float1 &x)
    {
        return mad(x, b - a, a);
    }

#endif

    static MATH_FORCEINLINE float4 lerp(const float4 &a, const float4 &b, const float1 &x)
    {
        return mad(x, b - a, a);
    }

    static MATH_FORCEINLINE float4 lerp(const float4 &a, const float4 &b, const float4 &x)
    {
        return mad(x, b - a, a);
    }

    static MATH_FORCEINLINE float3 lerp(const float3 &a, const float3 &b, const float3 &x)
    {
        return mad(x, b - a, a);
    }

    static MATH_FORCEINLINE float3 lerp(const float3 &a, const float3 &b, const float1 &x)
    {
        return mad(x, b - a, a);
    }

    static MATH_FORCEINLINE float2 lerp(const float2 &a, const float2 &b, const float2 &x)
    {
        return mad(x, b - a, a);
    }

    static MATH_FORCEINLINE float2 lerp(const float2 &a, const float2 &b, const float1 &x)
    {
        return mad(x, b - a, a);
    }

    //static MATH_FORCEINLINE float4 lerp(const float2 &a, const float2 &b, const float2 &x)
    //{
    //  return mad(x, b - a, a);
    //}

    static MATH_FORCEINLINE int compare_approx(float a, float b, float ep)
    {
        return (abs(b - a) <= ep);
    }

#if defined(MATH_HAS_NATIVE_SIMD) || defined(MATH_HAS_SIMD_FLOAT)
    static MATH_FORCEINLINE int1 compare_approx(const float1& a, const float1& b, const float1& ep)
    {
        return (abs(b - a) <= ep);
    }

#endif

    static MATH_FORCEINLINE int1 compare_approx(const float2& a, const float2& b, const float1& ep)
    {
        return (dot(b - a) <= ep * ep);
    }

    static MATH_FORCEINLINE int1 compare_approx(const float3& a, const float3& b, const float1& ep)
    {
        return (dot(b - a) <= ep * ep);
    }

    static MATH_FORCEINLINE int1 compare_approx(const float4& a, const float4& b, const float1& ep)
    {
        return (dot(b - a) <= ep * ep);
    }

    //
    //  extract: load a scalar float from a vector register, at the given index
    //  return value: the float value at the given index
    //
    static MATH_FORCEINLINE float extract(const float4 &v, unsigned int index)
    {
        return reinterpret_cast<const float*>(&v)[index];
    }

    static MATH_FORCEINLINE float extract(const float3 &v, unsigned int index)
    {
        return reinterpret_cast<const float*>(&v)[index];
    }

    static MATH_FORCEINLINE float extract(const float2 &v, unsigned int index)
    {
        return reinterpret_cast<const float*>(&v)[index];
    }

    //
    //  extract: load a scalar int from a vector register, at the given index
    //  return value: the int value at the given index
    //
    static MATH_FORCEINLINE int extract(const int4 &v, unsigned int index)
    {
        return reinterpret_cast<const int*>(&v)[index];
    }

    static MATH_FORCEINLINE int extract(const int3 &v, unsigned int index)
    {
        return reinterpret_cast<const int*>(&v)[index];
    }

    static MATH_FORCEINLINE int extract(const int2 &v, unsigned int index)
    {
        return reinterpret_cast<const int*>(&v)[index];
    }

    //
    //  insert: store a scalar float to a vector register, at the given index
    //
    static MATH_FORCEINLINE void insert(float4 &v, unsigned int index, float value)
    {
        reinterpret_cast<float*>(&v)[index] = value;
    }

    static MATH_FORCEINLINE void insert(float3 &v, unsigned int index, float value)
    {
        reinterpret_cast<float*>(&v)[index] = value;
    }

    static MATH_FORCEINLINE void insert(float2 &v, unsigned int index, float value)
    {
        reinterpret_cast<float*>(&v)[index] = value;
    }

    //
    //  insert: store a scalar int to a vector register, at the given index
    //
    static MATH_FORCEINLINE void insert(int4 &v, unsigned int index, int value)
    {
        reinterpret_cast<int*>(&v)[index] = value;
    }

    static MATH_FORCEINLINE void insert(int3 &v, unsigned int index, int value)
    {
        reinterpret_cast<int*>(&v)[index] = value;
    }

    static MATH_FORCEINLINE void insert(int2 &v, unsigned int index, int value)
    {
        reinterpret_cast<int*>(&v)[index] = value;
    }

    static MATH_FORCEINLINE float3 mirrorX(const float3& t)
    {
        return t * float3(-1.f, 1.f, 1.f);
    }

    static MATH_FORCEINLINE float4 vector(const float4& v)
    {
        return as_float4(as_int4(v) & int4(~0, ~0, ~0, 0));
    }

    ///////////////////////////////////////////////////////////
    //
    //  shiftRightLogical
    //  return value: a >> b (logical shift, operator>> uses arithmetic)
    //
    static MATH_FORCEINLINE int4 shiftRightLogical(const int4 &a, int b)
    {
#   if defined(__ARM_NEON)
        b = -b;
        return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32((int32x4_t)a), vld1q_dup_s32(&b)));
#   elif defined(__SSE__)
        return _mm_srli_epi32((__m128i)a, b);
#   else
        return int4((unsigned int)a.x >> b, (unsigned int)a.y >> b, (unsigned int)a.z >> b, (unsigned int)a.w >> b);
#   endif
    }

    ///////////////////////////////////////////////////////////
    //
    //  shiftLeftLogical
    //  return value: a << b (logical shift, operator<< uses arithmetic)
    //
    static MATH_FORCEINLINE int4 shiftLeftLogical(const int4 &a, int b)
    {
#   if defined(__ARM_NEON)
        return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32((int32x4_t)a), vld1q_dup_s32(&b)));
#   elif defined(__SSE__)
        return _mm_slli_epi32((__m128i)a, b);
#   else
        return int4((unsigned int)a.x << b, (unsigned int)a.y << b, (unsigned int)a.z << b, (unsigned int)a.w << b);
#   endif
    }

/*
#if defined(__clang__)
#   define _swiz_0   x
#   define _swiz_1   y
#   define _swiz_2   z
#   define _swiz_3   w
#   define swizzle_float4(v, x, y, z, w)    PP_CAT(v., PP_CAT(PP_CAT(PP_CAT(PP_CAT(_swiz_, x), PP_CAT(_swiz_, y)), PP_CAT(_swiz_, z)), PP_CAT(_swiz_, w)))
#   define swizzle_float3(v, x, y, z)       PP_CAT(v., PP_CAT(PP_CAT(PP_CAT(_swiz_, x), PP_CAT(_swiz_, y)), PP_CAT(_swiz_, z)))
#   define swizzle_float2(v, x, y)          PP_CAT(v., PP_CAT(PP_CAT(_swiz_, x), PP_CAT(_swiz_, y)))
#else
#   define swizzle_float4(v, x, y, z, w)    float4(v4f::SWIZ<SWZ(COMP_X + (x), COMP_X + (y), COMP_X + (z), COMP_X + (w))>::f((v4f::packed) (v)))
#   define swizzle_float3(v, x, y, z)       float3(v4f::SWIZ<SWZ(COMP_X + (x), COMP_X + (y), COMP_X + (z), COMP_N)>::f((v4f::packed) (v)))
#   define swizzle_float2(v, x, y)          float2(v4f::SWIZ<SWZ(COMP_X + (x), COMP_X + (y), COMP_N, COMP_N)>::f((v4f::packed) (v)))
#endif
*/
}

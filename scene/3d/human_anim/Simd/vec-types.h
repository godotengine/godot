#pragma once



#include "config.h"

#include "vec-intrin.h"

#include <cstring>

namespace math
{
    enum zero_t
    {
        ZERO = 0
    };
}

#if defined(MATH_HAS_NATIVE_SIMD)
namespace math
{
    typedef int int2 __attribute__ ((ext_vector_type(2)));
    typedef int int3 __attribute__ ((ext_vector_type(3)));
    typedef int int4 __attribute__ ((ext_vector_type(4)));

    typedef float float2 __attribute__ ((ext_vector_type(2)));
    typedef float float3 __attribute__ ((ext_vector_type(3)));
    typedef float float4 __attribute__ ((ext_vector_type(4)));

    template<typename T> static MATH_FORCEINLINE typename T::packed as_native(const T& v);

#if defined(__SSE__)
    static MATH_FORCEINLINE __m128i _mm_castpi64_pi128(int2 v)
    {
        int4 r = (int4) {v.x, v.y, r.z, r.w };
        return (__m128i)r;
    }

    static MATH_FORCEINLINE __m128 _mm_castps64_ps128(float2 v)
    {
        float4 r = (float4) {v.x, v.y, r.z, r.w };
        return (__m128)r;
    }

#endif

#if defined(MATH_HAS_SIMD_INT)

    struct int1
    {
#   if defined(__ARM_NEON)
        typedef int32x4_t   packed;
#   elif defined(__SSE__)
        typedef __m128i     packed;
#   elif MATH_HAS_SIMD_INT == 2
        typedef int2        packed;
#   else
        typedef int4        packed;
#   endif

#   if MATH_HAS_SIMD_INT == 2
        int2    p;
#   else
        int4    p;
#   endif

        MATH_EMPTYINLINE int1() {}

        MATH_FORCEINLINE int1(const int1 &v)
            :   p(v.p)
        {
        }

        MATH_FORCEINLINE int1(int x)
            :   p(x)
        {
        }

        MATH_FORCEINLINE int1(zero_t)
#if defined(__ARM_NEON)
            :   p(vdupq_n_s32(0))
#elif defined(__SSE__)
            :   p(_mm_setzero_si128())
#else
            :   p(0)
#endif
        {
        }

        MATH_FORCEINLINE explicit int1(packed p)
            :   p(p)
        {
        }

#   if defined(__ARM_NEON)
        MATH_FORCEINLINE explicit int1(int32x2_t p)
            :   p(vcombine_s32(p, p))
        {
        }

#   endif

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE operator int() const
        {
#   if defined(__SSE4_1__)
            return _mm_extract_epi32(p, 0);
#   else
            return p.x;
#   endif
        }

        MATH_FORCEINLINE operator int3() const
        {
            return p.xyz;
        }

        MATH_FORCEINLINE operator int4() const
        {
            return p;
        }

        MATH_FORCEINLINE int1 &operator=(int1 v)
        {
            p = v.p;
            return *this;
        }

        MATH_FORCEINLINE int1 &operator=(int x)
        {
            p = x;
            return *this;
        }
    };

#else

    typedef int int1;

#endif

    // clang cannot cast a int2/int3 to a native type of different size
#if defined(__clang__)
    static MATH_FORCEINLINE int1::packed as_native(const int3& v);
    static MATH_FORCEINLINE int1::packed as_native(const int2& v);
#endif

    static MATH_FORCEINLINE int2 int2_ctor(int x)
    {
        return (int2)x;
    }

    static MATH_FORCEINLINE int2 int2_ctor(int x, int y)
    {
        return (int2) {x, y };
    }

    static MATH_FORCEINLINE int2 int2_ctor(int2 v)
    {
        return v;
    }

#if defined(__ARM_NEON)

    static MATH_FORCEINLINE int2 int2_ctor(int32x2_t p)
    {
        return (int2)p;
    }

    static MATH_FORCEINLINE int2 int2_ctor(int32x4_t p)
    {
        return ((int4)p).xy;
    }

#elif defined(__SSE__)

    static MATH_FORCEINLINE int2 int2_ctor(__m128i p)
    {
        return ((int4)p).xy;
    }

#endif

    static MATH_FORCEINLINE int2 int2_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return int2_ctor(vdup_n_s32(0));
#elif defined(__SSE__)
        return int2_ctor(_mm_setzero_si128());
#else
        return int2_ctor(0);
#endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int2 int2_ctor(int1 v)
    {
        return v.p.xy;
    }

    static MATH_FORCEINLINE int2 int2_ctor(int1 x, int1 y)
    {
        return (int2) {x.p.x, y.p.x };
    }

#endif

    //
    // int3
    //
    static MATH_FORCEINLINE int3 int3_ctor(int x)
    {
        return (int3)x;
    }

    static MATH_FORCEINLINE int3 int3_ctor(int x, int y, int z)
    {
        return (int3) {x, y, z };
    }

    static MATH_FORCEINLINE int3 int3_ctor(int3 v)
    {
        return v;
    }

#if defined(__SSE2__)
    static MATH_FORCEINLINE int3 int3_ctor(__m128i p)
    {
        return ((int4)p).xyz;
    }

#endif

#if defined(__ARM_NEON)
    static MATH_FORCEINLINE int3 int3_ctor(int32x4_t p)
    {
        return ((int4)p).xyz;
    }

#endif

    static MATH_FORCEINLINE int3 int3_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return int3_ctor(vdupq_n_s32(0));
#elif defined(__SSE__)
        return int3_ctor(_mm_setzero_si128());
#else
        return int3_ctor(0);
#endif
    }

    static MATH_FORCEINLINE int3 int3_ctor(int2 xy, int z)
    {
#   if defined(__SSE4_1__)
        return int3_ctor(_mm_insert_epi32(_mm_castpi64_pi128(xy), z, 2));
#   else
        return (int3) {xy.x, xy.y, z };
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int3 int3_ctor(int1 v)
    {
        return v.p.xyz;
    }

    static MATH_FORCEINLINE int3 int3_ctor(int1 x, int1 y, int1 z)
    {
#   if defined(__SSE2__) && !TARGET_IPHONE_SIMULATOR && !TARGET_TVOS_SIMULATOR && !PLATFORM_OSX
        return ((int4)_mm_shuffle_ps((__m128)_mm_unpacklo_epi32(x.p, y.p), (__m128)_mm_unpacklo_epi32(z.p, z.p), _MM_SHUFFLE(1, 0, 1, 0))).xyz;
#   else
        return (int3) {x.p.x, y.p.x, z.p.x };
#   endif
    }

#endif

    //
    // int4
    //
    static MATH_FORCEINLINE int4 int4_ctor(int x)
    {
        return (int4)x;
    }

    static MATH_FORCEINLINE int4 int4_ctor(int x, int y, int z, int w)
    {
        return (int4) {x, y, z, w };
    }

    static MATH_FORCEINLINE int4 int4_ctor(int4 v)
    {
        return v;
    }

#if defined(__ARM_NEON)

    static MATH_FORCEINLINE int4 int4_ctor(int32x4_t p)
    {
        return p;
    }

#elif defined(__SSE2__)

    static MATH_FORCEINLINE int4 int4_ctor(__m128i p)
    {
        return p;
    }

#endif

    static MATH_FORCEINLINE int4 int4_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return int4_ctor(vdupq_n_s32(0));
#elif defined(__SSE__)
        return int4_ctor(_mm_setzero_si128());
#else
        return int4_ctor(0);
#endif
    }

    static MATH_FORCEINLINE int4 int4_ctor(int2 xy, int2 zw)
    {
#   if defined(__ARM_NEON)
        return vcombine_s32(xy, zw);
#   elif defined(__SSE__)
        return (__m128i)_mm_shuffle_ps((__m128)_mm_castpi64_pi128(xy), (__m128)_mm_castpi64_pi128(zw), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (int4) {xy.x, xy.y, zw.x, zw.y };
#   endif
    }

    static MATH_FORCEINLINE int4 int4_ctor(int3 xyz, int w)
    {
#   if defined(__ARM_NEON) && !defined(BUILD_HAS_VECTORS)
        return vsetq_lane_s32(w, as_native(xyz), 3);
#   elif defined(__SSE4_1__)
        return _mm_insert_epi32(as_native(xyz), w, 3);
#   else
        int4 r; r.xyz = xyz; r.w = w;
        return r;
#   endif
    }

#if defined(MATH_HAS_SIMD_INT)

    static MATH_FORCEINLINE int4 int4_ctor(int1 v)
    {
        return v.p;
    }

    static MATH_FORCEINLINE int4 int4_ctor(int1 x, int1 y, int1 z, int w)
    {
#   if defined(__SSE2__)
        return (__m128i)_mm_shuffle_ps((__m128)_mm_unpacklo_epi32(x.p, y.p), (__m128)_mm_unpacklo_epi32(z.p, _mm_cvtsi32_si128(w)), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (int4) {x.p.x, y.p.x, z.p.x, w };
#   endif
    }

    static MATH_FORCEINLINE int4 int4_ctor(int1 x, int1 y, int1 z, int1 w)
    {
#   if defined(__SSE2__)
        return (__m128i)_mm_shuffle_ps((__m128)_mm_unpacklo_epi32(x.p, y.p), (__m128)_mm_unpacklo_epi32(z.p, w.p), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (int4) {x.p.x, y.p.x, z.p.x, w.p.x };
#   endif
    }

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

    struct float1
    {
#   if defined(__ARM_NEON)
        typedef float32x4_t packed;
#   elif defined(__SSE__)
        typedef __m128      packed;
#   elif MATH_HAS_SIMD_FLOAT == 2
        typedef float2      packed;
#   else
        typedef float4      packed;
#   endif

#   if MATH_HAS_SIMD_FLOAT == 2
        float2  p;
#   else
        float4  p;
#   endif

        MATH_EMPTYINLINE float1() {}

        MATH_FORCEINLINE float1(const float1 &v)
            :   p(v.p)
        {
        }

        MATH_FORCEINLINE float1(float x)
            :   p(x)
        {
        }

        MATH_FORCEINLINE float1(zero_t)
#if defined(__ARM_NEON)
            :   p(vdupq_n_f32(0.f))
#elif defined(__SSE__)
            :   p(_mm_setzero_ps())
#else
            :   p(0)
#endif
        {
        }

        MATH_FORCEINLINE explicit float1(packed p)
            :   p(p)
        {
        }

#   if defined(__ARM_NEON)
        MATH_FORCEINLINE explicit float1(float32x2_t p)
            :   p(vcombine_f32(p, p))
        {
        }

#   endif

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE operator float() const
        {
#   if defined(__SSE__)
            return _mm_cvtss_f32(p);
#   else
            return p.x;
#   endif
        }

        MATH_FORCEINLINE operator float2() const
        {
            return p.xy;
        }

        MATH_FORCEINLINE operator float3() const
        {
            return p.xyz;
        }

        MATH_FORCEINLINE operator float4() const
        {
            return p;
        }

        MATH_FORCEINLINE float1 &operator=(float1 v)
        {
            p = v.p;
            return *this;
        }

        MATH_FORCEINLINE float1 &operator=(float x)
        {
            p = x;
            return *this;
        }
    };

#else

    typedef float float1;

#endif

    // clang cannot cast a float2/float3 to a native type of different size
#if defined(__clang__)
    static MATH_FORCEINLINE float1::packed as_native(const float3& v);
    static MATH_FORCEINLINE float1::packed as_native(const float2& v);
#endif

    //
    //  float2
    //
    static MATH_FORCEINLINE float2 float2_ctor(float x)
    {
        return (float2)x;
    }

    static MATH_FORCEINLINE float2 float2_ctor(float x, float y)
    {
        return (float2) {x, y };
    }

    static MATH_FORCEINLINE float2 float2_ctor(float2 v)
    {
        return v;
    }

#if defined(__ARM_NEON)

    static MATH_FORCEINLINE float2 float2_ctor(float32x2_t p)
    {
        return (float2)p;
    }

    static MATH_FORCEINLINE float2 float2_ctor(float32x4_t p)
    {
        return ((float4)p).xy;
    }

#elif defined(__SSE__)

    static MATH_FORCEINLINE float2 float2_ctor(__m128 p)
    {
        return ((float4)p).xy;
    }

#endif

    static MATH_FORCEINLINE float2 float2_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return float2_ctor(vdup_n_f32(0.f));
#elif defined(__SSE__)
        return float2_ctor(_mm_setzero_ps());
#else
        return float2_ctor(0.f);
#endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float2 float2_ctor(float1 v)
    {
        return v.p.xy;
    }

    static MATH_FORCEINLINE float2 float2_ctor(float1 x, float1 y)
    {
        return (float2) {x.p.x, y.p.x };
    }

#endif

    //
    //  float3
    //
    static MATH_FORCEINLINE float3 float3_ctor(float x)
    {
        return (float3)x;
    }

    static MATH_FORCEINLINE float3 float3_ctor(float x, float y, float z)
    {
        return (float3) {x, y, z };
    }

    static MATH_FORCEINLINE float3 float3_ctor(float3 v)
    {
        return v;
    }

#if defined(__SSE2__)
    static MATH_FORCEINLINE float3 float3_ctor(__m128 p)
    {
        return ((float4)p).xyz;
    }

#endif

#if defined(__ARM_NEON)
    static MATH_FORCEINLINE float3 float3_ctor(float32x4_t p)
    {
        return ((float4)p).xyz;
    }

#endif

    static MATH_FORCEINLINE float3 float3_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return float3_ctor(vdupq_n_f32(0.f));
#elif defined(__SSE__)
        return float3_ctor(_mm_setzero_ps());
#else
        return float3_ctor(0.f);
#endif
    }

    static MATH_FORCEINLINE float3 float3_ctor(float2 xy, float z)
    {
#   if defined(__SSE4_1__)
        return float3_ctor(_mm_insert_ps(_mm_castps64_ps128(xy), _mm_cvtf32_ss(z), 0x28));
#   else
        float3 r; r.xy = xy; r.z = z;
        return r;
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float3 float3_ctor(float1 v)
    {
        return v.p.xyz;
    }

    static MATH_FORCEINLINE float3 float3_ctor(float1 x, float1 y, float1 z)
    {
#   if defined(__SSE__) && !TARGET_IPHONE_SIMULATOR && !TARGET_TVOS_SIMULATOR && !PLATFORM_OSX
        return ((float4)_mm_shuffle_ps(_mm_unpacklo_ps(x.p, y.p), _mm_unpacklo_ps(z.p, z.p), _MM_SHUFFLE(1, 0, 1, 0))).xyz;
#   else
        return (float3) {x.p.x, y.p.x, z.p.x };
#   endif
    }

#endif

    //
    //  float4
    //
    static MATH_FORCEINLINE float4 float4_ctor(float x)
    {
        return (float4)x;
    }

    static MATH_FORCEINLINE float4 float4_ctor(float x, float y, float z, float w)
    {
        return (float4) {x, y, z, w };
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 float4_ctor(float1 x, float y, float z, float w)
    {
        return (float4) {x, y, z, w };
    }

    static MATH_FORCEINLINE float4 float4_ctor(float1 x, float y, float z, float1 w)
    {
        return (float4) {x, y, z, w };
    }

    static MATH_FORCEINLINE float4 float4_ctor(float x, float y, float z, float1 w)
    {
        return (float4) {x, y, z, w };
    }

#endif

    static MATH_FORCEINLINE float4 float4_ctor(float4 v)
    {
        return v;
    }

#if defined(__ARM_NEON)

    static MATH_FORCEINLINE float4 float4_ctor(float32x4_t p)
    {
        return p;
    }

#elif defined(__SSE2__)

    static MATH_FORCEINLINE float4 float4_ctor(__m128 p)
    {
        return p;
    }

#endif

    static MATH_FORCEINLINE float4 float4_ctor(zero_t)
    {
#if defined(__ARM_NEON)
        return float4_ctor(vdupq_n_f32(0.f));
#elif defined(__SSE__)
        return float4_ctor(_mm_setzero_ps());
#else
        return float4_ctor(0.f);
#endif
    }

    static MATH_FORCEINLINE float4 float4_ctor(float2 xy, float2 zw)
    {
#   if defined(__ARM_NEON)
        return vcombine_f32(xy, zw);
#   elif defined(__SSE__)
        return _mm_shuffle_ps(_mm_castps64_ps128(xy), _mm_castps64_ps128(zw), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (float4) {xy.x, xy.y, zw.x, zw.y };
#   endif
    }

    static MATH_FORCEINLINE float4 float4_ctor(float3 xyz, float w)
    {
#   if defined(__ARM_NEON) && defined(__APPLE__)
        return vsetq_lane_f32(w, as_native(xyz), 3);
#   elif defined(__SSE4_1__)
        return _mm_insert_ps(as_native(xyz), _mm_cvtf32_ss(w), 0x30);
#   else
        float4 r; r.xyz = xyz; r.w = w;
        return r;
#   endif
    }

#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float4 float4_ctor(float1 v)
    {
        return v.p;
    }

    static MATH_FORCEINLINE float4 float4_ctor(float1 x, float1 y, float1 z, float w)
    {
#   if defined(__SSE__)
        return _mm_shuffle_ps(_mm_unpacklo_ps(x.p, y.p), _mm_unpacklo_ps(z.p, _mm_cvtf32_ss(w)), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (float4) {x.p.x, y.p.x, z.p.x, w };
#   endif
    }

    static MATH_FORCEINLINE float4 float4_ctor(float1 x, float1 y, float1 z, float1 w)
    {
#   if defined(__SSE__)
        return _mm_shuffle_ps(_mm_unpacklo_ps(x.p, y.p), _mm_unpacklo_ps(z.p, w.p), _MM_SHUFFLE(1, 0, 1, 0));
#   else
        return (float4) {x.p.x, y.p.x, z.p.x, w.p.x };
#   endif
    }

#endif

#define MATH_HAS_CONSTRUCTOR_DEFINE
#   define int3(...)   int3_ctor(__VA_ARGS__)
#   define int2(...)   int2_ctor(__VA_ARGS__)
#   define int4(...)   int4_ctor(__VA_ARGS__)

#   define float3(...) float3_ctor(__VA_ARGS__)
#   define float2(...) float2_ctor(__VA_ARGS__)
#   define float4(...) float4_ctor(__VA_ARGS__)

    // clang cannot cast a float2/float3 to a native type of different size
#if defined(__clang__)
    static MATH_FORCEINLINE float1::packed as_native(const float2& v)
    {
        float4 r = (float4) {v.x, v.y, r.z, r.w };
        return r;
    }

    static MATH_FORCEINLINE float1::packed as_native(const float3& v)
    {
        float4 r = (float4) {v.x, v.y, v.z, r.w };
        return r;
    }

    static MATH_FORCEINLINE int1::packed as_native(const int3& v)
    {
        int4 r = (int4) {v.x, v.y, v.z, r.w };
        return r;
    }

    static MATH_FORCEINLINE int1::packed as_native(const int2& v)
    {
        int4 r = (int4) {v.x, v.y, r.z, r.w };
        return r;
    }

#else
    template<typename T> static MATH_FORCEINLINE typename T::packed as_native(const T& v)
    {
        typedef typename T::packed packed_t;
        return (packed_t)v;
    }

#endif
}

#include "vec-oper.h"

#elif defined(MATH_HAS_SIMD_FLOAT)

#include "vec-simd.h"

#if defined(__ARM_NEON)
#   include "vec-neon.h"
#elif defined(__SSE__)
#   include "vec-sse.h"
#endif

#include "vec-oper.h"

namespace math
{
    typedef struct _float1 : meta::lv<meta::v4f, meta::sp<meta::v4f, meta::SWZ_ANY, 1>, 1>
    {
        typedef meta::v4f  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_ANY, 1>, 1> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _float1() {}

        MATH_FORCEINLINE _float1(const packed &p)
        {
            this->p = p;
        }

#   if defined(__ARM_NEON)
        MATH_FORCEINLINE _float1(const float32x2_t &p)
        {
            this->p = vcombine_f32(p, p);
        }

#   endif
        MATH_FORCEINLINE _float1(const _float1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float1(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _float1(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename RHS> MATH_FORCEINLINE _float1(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 4)>::f(rhs.p);
        }

        MATH_FORCEINLINE const _float1 &operator=(const _float1 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }
    } vec_attr float1;

    typedef struct _float2 : meta::lv<meta::v4f, meta::sp<meta::v4f, meta::SWZ_XY, 2>, 2>
    {
        typedef meta::v4f  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XY, 2>, 2> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _float2() {}

        MATH_FORCEINLINE _float2(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _float2(const float1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float2(const _float2 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float2(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _float2(type x, type y)
        {
            this->p = T::CTOR(x, y);
        }

        MATH_FORCEINLINE _float2(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y> MATH_FORCEINLINE _float2(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, 0, 0>::f(x.p, y.p, x.p, y.p);
        }

        template<typename RHS> MATH_FORCEINLINE _float2(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 2)>::f(rhs.p);
        }

        template<typename RHS> MATH_FORCEINLINE _float2(const meta::v<T, RHS, 2> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _float2 &operator=(const _float2 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _float2 *operator&() const
        {
            return (_float2 *) this;
        }
    } vec_attr float2;

    typedef struct _float3 : meta::lv<meta::v4f, meta::sp<meta::v4f, meta::SWZ_XYZ, 3>, 3>
    {
        typedef meta::v4f  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XYZ, 3>, 3> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _float3() {}

        MATH_FORCEINLINE _float3(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _float3(const float1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float3(const _float3 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float3(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _float3(type x, type y, type z)
        {
            this->p = T::CTOR(x, y, z);
        }

        MATH_FORCEINLINE _float3(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y, typename Z> MATH_FORCEINLINE _float3(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, 0>::f(x.p, y.p, z.p, z.p);
        }

        template<typename RHS> MATH_FORCEINLINE _float3(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 3)>::f(rhs.p);
        }

        template<typename XY, typename Z> MATH_FORCEINLINE _float3(const meta::v<T, XY, 2> &xy, const meta::v<T, Z, 1> &z)
        {
            this->p = T::MASK<XY::SWZ, (Z::SWZ << 8) & meta::MSK_Z, meta::MSK_Z>::f(xy.p, z.p);
        }

        template<typename XY> MATH_FORCEINLINE _float3(const meta::v<T, XY, 2> &xy, type z)
        {
            this->p = T::MASK<XY::SWZ, meta::SWZ_ANY, meta::MSK_Z>::f(xy.p, T::CTOR(z));
        }

        template<typename RHS> MATH_FORCEINLINE _float3(const meta::v<T, RHS, 3> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _float3 &operator=(const _float3 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _float3 *operator&() const
        {
            return (_float3 *) this;
        }
    } vec_attr float3;

    typedef struct _float4 : meta::lv<meta::v4f, meta::sp<meta::v4f, meta::SWZ_XYZW, 4>, 4>
    {
        typedef meta::v4f  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XYZW, 4>, 4> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _float4() {}

        MATH_FORCEINLINE _float4(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _float4(const float1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float4(const _float4 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _float4(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _float4(type x, type y, type z, type w)
        {
            this->p = T::CTOR(x, y, z, w);
        }

        MATH_FORCEINLINE _float4(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y, typename Z, typename W> MATH_FORCEINLINE _float4(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z, const meta::v<T, W, 1> &w)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, W::SWZ>::f(x.p, y.p, z.p, w.p);
        }

        template<typename X, typename Y, typename Z> MATH_FORCEINLINE _float4(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z, type w)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, meta::SWZ_ANY>::f(x.p, y.p, z.p, T::CTOR(w));
        }

        template<typename RHS> MATH_FORCEINLINE _float4(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 4)>::f(rhs.p);
        }

        template<typename XY, typename ZW> MATH_FORCEINLINE _float4(const meta::v<T, XY, 2> &xy, const meta::v<T, ZW, 2> &zw)
        {
            this->p = T::MASK<XY::SWZ, (ZW::SWZ << 8) & meta::MSK_ZW, meta::MSK_ZW>::f(xy.p, zw.p);
        }

        template<typename XYZ, typename W> MATH_FORCEINLINE _float4(const meta::v<T, XYZ, 3> &xyz, const meta::v<T, W, 1> &w)
        {
            this->p = T::MASK<XYZ::SWZ, (W::SWZ << 12) & meta::MSK_W, meta::MSK_W>::f(xyz.p, w.p);
        }

        template<typename XYZ> MATH_FORCEINLINE _float4(const meta::v<T, XYZ, 3> &xyz, type w)
        {
            this->p = T::SET<SWZ(meta::COMP_W, 0, 0, 0)>::f(T::SWIZ<XYZ::SWZ>::f(xyz.p), w);
        }

        template<typename RHS> MATH_FORCEINLINE _float4(const meta::v<T, RHS, 4> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _float4 &operator=(const _float4 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _float4 *operator&() const
        {
            return (_float4 *) this;
        }
    } vec_attr float4;

    typedef struct _int1 : meta::lv<meta::v4i, meta::sp<meta::v4i, meta::SWZ_ANY, 1>, 1>
    {
        typedef meta::v4i  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_ANY, 1>, 1> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _int1() {}

        MATH_FORCEINLINE _int1(const packed &p)
        {
            this->p = p;
        }

#   if defined(__ARM_NEON)
        MATH_FORCEINLINE _int1(const int32x2_t &p)
        {
            this->p = vcombine_s32(p, p);
        }

#   endif
        MATH_FORCEINLINE _int1(const _int1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int1(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _int1(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename RHS> MATH_FORCEINLINE _int1(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 4)>::f(rhs.p);
        }

        const _int1 &operator=(const _int1 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }
    } vec_attr int1;

    typedef struct _int2 : meta::lv<meta::v4i, meta::sp<meta::v4i, meta::SWZ_XY, 2>, 2>
    {
        typedef meta::v4i  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XY, 2>, 2> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _int2() {}

        MATH_FORCEINLINE _int2(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _int2(const int1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int2(const _int2 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int2(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _int2(type x, type y)
        {
            this->p = T::CTOR(x, y);
        }

        MATH_FORCEINLINE _int2(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y> MATH_FORCEINLINE _int2(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, 0, 0>::f(x.p, y.p, x.p, y.p);
        }

        template<typename RHS> MATH_FORCEINLINE _int2(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 2)>::f(rhs.p);
        }

        template<typename RHS> MATH_FORCEINLINE _int2(const meta::v<T, RHS, 2> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _int2 &operator=(const _int2 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _int2 *operator&() const
        {
            return (_int2 *) this;
        }
    } vec_attr int2;

    typedef struct _int3 : meta::lv<meta::v4i, meta::sp<meta::v4i, meta::SWZ_XYZ, 3>, 3>
    {
        typedef meta::v4i  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XYZ, 3>, 3> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _int3() {}

        MATH_FORCEINLINE _int3(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _int3(const int1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int3(const _int3 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int3(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _int3(type x, type y, type z)
        {
            this->p = T::CTOR(x, y, z);
        }

        MATH_FORCEINLINE _int3(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y, typename Z> MATH_FORCEINLINE _int3(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, 0>::f(x.p, y.p, z.p, z.p);
        }

        template<typename RHS> MATH_FORCEINLINE _int3(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 4)>::f(rhs.p);
        }

        template<typename XY, typename Z> MATH_FORCEINLINE _int3(const meta::v<T, XY, 2> &xy, const meta::v<T, Z, 1> &z)
        {
            this->p = T::MASK<XY::SWZ, (Z::SWZ << 8) & meta::MSK_Z, meta::MSK_Z>::f(xy.p, z.p);
        }

        template<typename RHS> MATH_FORCEINLINE _int3(const meta::v<T, RHS, 3> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _int3 &operator=(const _int3 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _int3 *operator&() const
        {
            return (_int3 *) this;
        }
    } vec_attr int3;

    typedef struct _int4 : meta::lv<meta::v4i, meta::sp<meta::v4i, meta::SWZ_XYZW, 4>, 4>
    {
        typedef meta::v4i  T;
        typedef meta::lv<T, meta::sp<T, meta::SWZ_XYZW, 4>, 4> base;
        typedef T::type  type;
        typedef T::packed packed;

        MATH_EMPTYINLINE _int4() {}

        MATH_FORCEINLINE _int4(const packed &p)
        {
            this->p = p;
        }

        MATH_FORCEINLINE _int4(const int1 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int4(const _int4 &v)
        {
            this->p = v.p;
        }

        MATH_FORCEINLINE _int4(type x)
        {
            this->p = T::CTOR(x);
        }

        MATH_FORCEINLINE _int4(type x, type y, type z, type w)
        {
            this->p = T::CTOR(x, y, z, w);
        }

        MATH_FORCEINLINE _int4(zero_t)
        {
            this->p = T::ZERO();
        }

        template<typename X, typename Y, typename Z, typename W> MATH_FORCEINLINE _int4(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z, const meta::v<T, W, 1> &w)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, W::SWZ>::f(x.p, y.p, z.p, w.p);
        }

        template<typename X, typename Y, typename Z> MATH_FORCEINLINE _int4(const meta::v<T, X, 1> &x, const meta::v<T, Y, 1> &y, const meta::v<T, Z, 1> &z, type w)
        {
            this->p = T::GATHER<X::SWZ, Y::SWZ, Z::SWZ, meta::SWZ_ANY>::f(x.p, y.p, z.p, T::CTOR(w));
        }

        template<typename RHS> MATH_FORCEINLINE _int4(const meta::v<T, RHS, 1> &rhs)
        {
            this->p = T::SWIZ<BC(RHS::SWZ, 4)>::f(rhs.p);
        }

        template<typename XY, typename ZW> MATH_FORCEINLINE _int4(const meta::v<T, XY, 2> &xy, const meta::v<T, ZW, 2> &zw)
        {
            this->p = T::MASK<XY::SWZ, (ZW::SWZ << 8) & meta::MSK_ZW, meta::MSK_ZW>::f(xy.p, zw.p);
        }

        template<typename RHS> MATH_FORCEINLINE _int4(const meta::v<T, RHS, 4> &rhs)
        {
            this->p = T::SWIZ<RHS::SWZ>::f(rhs.p);
        }

        MATH_FORCEINLINE const _int4 &operator=(const _int4 &v) const
        {
            this->p = v.p;
            return *this;
        }

        MATH_FORCEINLINE explicit_operator packed() const
        {
            return p;
        }

        MATH_FORCEINLINE _int4 *operator&() const
        {
            return (_int4 *) this;
        }
    } vec_attr int4;


    template<typename T> static MATH_FORCEINLINE typename T::packed as_native(const T& v)
    {
        typedef typename T::packed packed_t;
        return (packed_t)v;
    }
}

#else

#include "vec-generic.h"
#include "vec-oper.h"

namespace math
{
    typedef int     int1;
    typedef float   float1;

    DECL_VEC2(int2, int)
    DECL_VEC3(int3, int)
    DECL_VEC4(int4, int)

    DECL_VEC2(float2, float)
    DECL_VEC3(float3, float)
    DECL_VEC4(float4, float)

    static MATH_FORCEINLINE float4 as_native(const float3& v)
    {
        return float4(v, 0);
    }

    static MATH_FORCEINLINE float4 as_native(const float2& v)
    {
        return float4(v, float2(0));
    }

    static MATH_FORCEINLINE int4 as_native(const int3& v)
    {
        return int4(v, 0);
    }

    static MATH_FORCEINLINE int4 as_native(const int2& v)
    {
        return int4(v, int2(0));
    }
}

#endif

#if defined(MATH_HAS_SIMD_FLOAT)

namespace math
{
    /*
     *  vector loading and storing functions (pointer is assumed to be float-aligned, but not necessarily vector-aligned)
     */

    static MATH_FORCEINLINE float4 vload4f(const float* p)
    {
#   if defined(__ARM_NEON)
        return float4(vld1q_f32(p));
#   elif defined(__SSE__)
        return float4(_mm_loadu_ps(p));
#   else
        return float4(p[0], p[1], p[2], p[3]);
#   endif
    }

    static MATH_FORCEINLINE void vstore4f(float* p, const float4& v)
    {
#   if defined(__ARM_NEON)
        vst1q_f32(p, (float32x4_t)v);
#   elif defined(__SSE__)
        _mm_storeu_ps(p, (__m128)v);
#   else
        p[0] = (float)v.x;
        p[1] = (float)v.y;
        p[2] = (float)v.z;
        p[3] = (float)v.w;
#   endif
    }

    /*
     *  vector loading and storing functions (pointer is assumed to be vector-aligned)
     */

    static MATH_FORCEINLINE float4 vload4f_aligned(const float* p)
    {
#   if defined(__ARM_NEON)
        return float4(vld1q_f32(p));
#   elif defined(__SSE__)
        return float4(_mm_load_ps(p));
#   else
        return float4(p[0], p[1], p[2], p[3]);
#   endif
    }

    static MATH_FORCEINLINE void vstore4f_aligned(float* p, const float4& v)
    {
#   if defined(__ARM_NEON)
        vst1q_f32(p, (float32x4_t)v);
#   elif defined(__SSE__)
        _mm_store_ps(p, (__m128)v);
#   else
        p[0] = (float)v.x;
        p[1] = (float)v.y;
        p[2] = (float)v.z;
        p[3] = (float)v.w;
#   endif
    }

    static MATH_FORCEINLINE float3 vload3f(const float* p)
    {
#   if defined(__ARM_NEON)
        float32x2_t lo = vld1_f32(p);
        float32x2_t hi = vld1_dup_f32(p + 2);
        return float3(vcombine_f32(lo, hi));
#   elif defined(__SSE2__)
        return float3(_mm_movelh_ps(_mm_castpd_ps(_mm_load_sd((const double*)p)), _mm_load_ss(p + 2)));
#   else
        return float3(p[0], p[1], p[2]);
#   endif
    }

    static MATH_FORCEINLINE void vstore3f(float* p, const float3& v)
    {
// @TODO: this makes most mesh collider runtime tests to fail on windows 32 standalone release
// @TODO2: looks like SSE2 path is way slower than fallback path. Do not re-activate without validating perf
//#   elif defined (__SSE2__)
//      __m128i n = _mm_set_epi32 (0, ~0, ~0, ~0);
//      float4 u; u.xyz = v;
//      _mm_maskmoveu_si128 (_mm_castps_si128 ((__m128) u), n, (char *) p);
//#   else
        p[0] = (float)v.x;
        p[1] = (float)v.y;
        p[2] = (float)v.z;
//#   endif
    }

    static MATH_FORCEINLINE float2 vload2f(const float* p)
    {
#   if defined(__ARM_NEON)
#       if defined(MATH_HAS_NATIVE_SIMD)
        return float2(vld1_f32(p));
#       else
        float32x2_t v = vld1_f32(p);
        return float2(vcombine_f32(v, v));
#       endif
#   elif defined(__SSE2__)
        float4 v4 = float4(_mm_castpd_ps(_mm_load_sd((const double*)p)));
        return v4.xy;
#   else
        return float2(p[0], p[1]);
#   endif
    }

    static MATH_FORCEINLINE void vstore2f(float* p, const float2& v)
    {
#   if defined(__ARM_NEON)
#       if defined(MATH_HAS_NATIVE_SIMD)
        vst1_f32(p, (float32x2_t)v);
#       else
        vst1_f32(p, vget_low_f32((float32x4_t)v));
#       endif
#   elif defined(__SSE2__)
#       if defined(MATH_HAS_NATIVE_SIMD)
        float4 v4 = (float4) {v.x, v.y, v4.z, v4.w };
        _mm_store_sd((double*)p, _mm_castps_pd((__m128)v4));
#       else
        _mm_store_sd((double*)p, _mm_castps_pd((__m128)v));
#       endif
#   else
        p[0] = (float)v.x;
        p[1] = (float)v.y;
#   endif
    }

    static MATH_FORCEINLINE float1 vload1f(const float* p)
    {
#   if defined(__ARM_NEON)
        return float1(vld1q_dup_f32(p));
#   elif defined(__AVX__)
        return float1(_mm_broadcast_ss(p));
#   elif defined(__SSE__)
        return float1(_mm_load1_ps(p));
#   else
        return float1(p[0]);
#   endif
    }

    static MATH_FORCEINLINE void vstore1f(float* p, const float1& v)
    {
#   if defined(__ARM_NEON)
        vst1q_lane_f32(p, (float32x4_t)v, 0);
#   elif defined(__SSE2__)
        _mm_store_ss(p, (__m128)v);
#   else
        p[0] = (float)v;
#endif
    }

    static MATH_FORCEINLINE int4 vload4i(const int* p)
    {
#   if defined(__ARM_NEON)
        return int4(vld1q_s32(p));
#   elif defined(__SSE__)
        return int4(_mm_loadu_si128((const __m128i*)p));
#   else
        return int4(p[0], p[1], p[2], p[3]);
#   endif
    }

    static MATH_FORCEINLINE int4 vload4i_aligned(const int* p)
    {
#   if defined(__ARM_NEON)
        return int4(vld1q_s32(p));
#   elif defined(__SSE__)
        return int4(_mm_load_si128((const __m128i*)p));
#   else
        return int4(p[0], p[1], p[2], p[3]);
#   endif
    }

    static MATH_FORCEINLINE void vstore4i(int* p, const int4& v)
    {
#   if defined(__ARM_NEON)
        vst1q_s32(p, (int32x4_t)v);
#   elif defined(__SSE__)
        _mm_storeu_si128((__m128i*)p, (__m128i)v);
#   else
        p[0] = (int)v.x;
        p[1] = (int)v.y;
        p[2] = (int)v.z;
        p[3] = (int)v.w;
#   endif
    }

    static MATH_FORCEINLINE void vstore4i_aligned(int* p, const int4& v)
    {
#   if defined(__ARM_NEON)
        vst1q_s32(p, (int32x4_t)v);
#   elif defined(__SSE__)
        _mm_store_si128((__m128i*)p, (__m128i)v);
#   else
        p[0] = (int)v.x;
        p[1] = (int)v.y;
        p[2] = (int)v.z;
        p[3] = (int)v.w;
#   endif
    }

    static MATH_FORCEINLINE int3 vload3i(const int* p)
    {
        return int3(p[0], p[1], p[2]);
    }

    static MATH_FORCEINLINE void vstore3i(int* p, const int3& v)
    {
        p[0] = (int)v.x;
        p[1] = (int)v.y;
        p[2] = (int)v.z;
    }

    static MATH_FORCEINLINE int2 vload2i(const int* p)
    {
        return int2(p[0], p[1]);
    }

    static MATH_FORCEINLINE void vstore2i(int* p, const int2& v)
    {
        p[0] = (int)v.x;
        p[1] = (int)v.y;
    }

    static MATH_FORCEINLINE int1 vload1i(const int* p)
    {
#   if defined(__ARM_NEON)
        return int1(vld1q_dup_s32(p));
#   elif defined(__AVX__)
        return int1(_mm_castps_si128(_mm_broadcast_ss((float*)p)));
#   elif defined(__SSE__)
        return int1(_mm_castps_si128(_mm_load1_ps((float*)p)));
#   else
        return int1(p[0]);
#   endif
    }

    static MATH_FORCEINLINE void vstore1i(int* p, const int1& v)
    {
#   if defined(__ARM_NEON)
        vst1q_lane_s32(p, (int32x4_t)v, 0);
#   elif defined(__SSE2__)
        _mm_store_ss((float*)p, _mm_castsi128_ps((__m128i)v));
#   else
        p[0] = (int)v;
#endif
    }

    static MATH_FORCEINLINE int4 vload16c(const char* p)
    {
#   if defined(__ARM_NEON)
        return vreinterpretq_s32_s8(vld1q_s8(reinterpret_cast<const signed char*>(p)));
#   elif defined(__SSE__)
        return int4(_mm_loadu_si128((const __m128i*)p));
#   else
        int data[4];
        std::memcpy(data, p, 16);
        return int4(data[0], data[1], data[2], data[3]);
#   endif
    }

    static MATH_FORCEINLINE int4 vload16uc(const unsigned char* p)
    {
        return vload16c(reinterpret_cast<const char*>(p));
    }

    static MATH_FORCEINLINE int1 vload4c(const char* p)
    {
#if defined(__ARM_NEON)
        // note that p may not be aligned and vld1q_dup_s32 which vloadi uses
        // expects 32-bit aligned pointer. We can't use `data = *reinterpret_cast<const int*>(p);`
        // because this is undefined behavior will result in crashes on certain platforms.
        int data;
        std::memcpy(&data, p, 4);
        return int1(data);
#else
        int data;
        std::memcpy(&data, p, 4);
        return vload1i(&data);
#endif
    }

    static MATH_FORCEINLINE void vstore16c(char* p, const int4& v)
    {
#if defined(__ARM_NEON)
        vst1q_s8(reinterpret_cast<signed char*>(p), vreinterpretq_s8_s32((int32x4_t)v));
#elif defined(__SSE__)
        _mm_storeu_si128((__m128i*)p, (__m128i)v);
#else
        int data[4];
        data[0] = (int)v.x;
        data[1] = (int)v.y;
        data[2] = (int)v.z;
        data[3] = (int)v.w;
        std::memcpy(p, &data, 16);
#endif
    }

    static MATH_FORCEINLINE void vstore4c(char* p, const int1& v)
    {
#if defined(__ARM_NEON)
        // note that p may not be aligned and vst1q_lane_s32 which vstore1i uses
        // expects 32-bit aligned pointer. We can't use `*reinterpret_cast<const int*>(p) = data`
        // because this is undefined behavior will result in crashes on certain platforms.
        int data;
#if defined(MATH_HAS_NATIVE_SIMD)
        vst1q_lane_s32(&data, v, 0);
#else
        vst1q_lane_s32(&data, v.p, 0);
#endif
        std::memcpy(p, &data, 4);
#elif defined(__SSE2__)
        _mm_store_ss(reinterpret_cast<float*>(p), _mm_castsi128_ps((__m128i)v));
#else
        int data = (int)v;
        std::memcpy(p, &data, 4);
#endif
    }

    // same as vstore4f_aligned, but bypasses cache. Faster for write-only data. Remember to call vstream_fence before reading the data!
    static MATH_FORCEINLINE void vstream4f_aligned(float* p, const float4& v)
    {
#   if defined(__SSE__)
        _mm_stream_ps((float*)p, (__m128)v);
#   else
        vstore4f_aligned(p, v);
#   endif
    }

    // same as vstore4i_aligned, but bypasses cache. Faster for write-only data. Remember to call vstream_fence before reading the data!
    static MATH_FORCEINLINE void vstream4i_aligned(int* p, const int4& v)
    {
#   if defined(__SSE__)
        _mm_stream_si128((__m128i*)p, (__m128i)v);
#   else
        vstore4i_aligned(p, v);
#   endif
    }

    // ensure all data written by vstream4X_aligned is globally visible before any subsequent load instructions
    static MATH_FORCEINLINE void vstream_fence()
    {
#   if defined(__SSE__)
        _mm_sfence();
#   endif
    }
}

#else

namespace math
{
    static MATH_FORCEINLINE float4 vload4f(const float* p)
    {
        return float4(p[0], p[1], p[2], p[3]);
    }

    static MATH_FORCEINLINE void vstore4f(float* p, const float4& v)
    {
        p[0] = (float)v.x;
        p[1] = (float)v.y;
        p[2] = (float)v.z;
        p[3] = (float)v.w;
    }

    static MATH_FORCEINLINE float4 vload4f_aligned(const float* p)
    {
        return vload4f(p);
    }

    static MATH_FORCEINLINE void vstore4f_aligned(float* p, const float4& v)
    {
        vstore4f(p, v);
    }

    static MATH_FORCEINLINE float3 vload3f(const float* p)
    {
        return float3(p[0], p[1], p[2]);
    }

    static MATH_FORCEINLINE void vstore3f(float* p, const float3& v)
    {
        p[0] = (float)v.x;
        p[1] = (float)v.y;
        p[2] = (float)v.z;
    }

    static MATH_FORCEINLINE float2 vload2f(const float* p)
    {
        return float2(p[0], p[1]);
    }

    static MATH_FORCEINLINE void vstore2f(float* p, const float2& v)
    {
        p[0] = (float)v.x;
        p[1] = (float)v.y;
    }

    static MATH_FORCEINLINE float1 vload1f(const float* p)
    {
        return float1(p[0]);
    }

    static MATH_FORCEINLINE void vstore1f(float* p, const float1& v)
    {
        p[0] = (float)v;
    }

    static MATH_FORCEINLINE int4 vload4i(const int* p)
    {
        return int4(p[0], p[1], p[2], p[3]);
    }

    static MATH_FORCEINLINE int4 vload4i_aligned(const int* p)
    {
        return vload4i(p);
    }

    static MATH_FORCEINLINE void vstore4i(int* p, const int4& v)
    {
        p[0] = (int)v.x;
        p[1] = (int)v.y;
        p[2] = (int)v.z;
        p[3] = (int)v.w;
    }

    static MATH_FORCEINLINE void vstore4i_aligned(int* p, const int4& v)
    {
        vstore4i(p, v);
    }

    static MATH_FORCEINLINE int3 vload3i(const int* p)
    {
        return int3(p[0], p[1], p[2]);
    }

    static MATH_FORCEINLINE void vstore3i(int* p, const int3& v)
    {
        p[0] = (int)v.x;
        p[1] = (int)v.y;
        p[2] = (int)v.z;
    }

    static MATH_FORCEINLINE int2 vload2i(const int* p)
    {
        return int2(p[0], p[1]);
    }

    static MATH_FORCEINLINE void vstore2i(int* p, const int2& v)
    {
        p[0] = (int)v.x;
        p[1] = (int)v.y;
    }

    static MATH_FORCEINLINE int1 vload1i(const int* p)
    {
        return int1(p[0]);
    }

    static MATH_FORCEINLINE void vstore1i(int* p, const int1& v)
    {
        p[0] = (int)v;
    }

    static MATH_FORCEINLINE int4 vload16uc(const unsigned char* p)
    {
        int i[4];
        std::memcpy(i, p, 16);
        return vload4i(i);
    }

    static MATH_FORCEINLINE int4 vload16c(const char* p)
    {
        int i[4];
        std::memcpy(i, p, 16);
        return vload4i(i);
    }

    static MATH_FORCEINLINE void vstore16c(char* p, const int4& v)
    {
        int i[4];
        vstore4i(i, v);
        std::memcpy(p, i, 16);
    }

    static MATH_FORCEINLINE int1 vload4c(const char* p)
    {
        int i[1];
        std::memcpy(i, p, 4);
        return vload1i(i);
    }

    static MATH_FORCEINLINE void vstore4c(char* p, const int1& v)
    {
        int i = (int)v;
        std::memcpy(p, &i, 4);
    }

    static MATH_FORCEINLINE void vstream4f_aligned(float* p, const float4& v)
    {
        vstore4f_aligned(p, v);
    }

    static MATH_FORCEINLINE void vstream4i_aligned(int* p, const int4& v)
    {
        vstore4i_aligned(p, v);
    }

    static MATH_FORCEINLINE void vstream_fence()
    {
    }
}

#endif

namespace math
{
#   ifdef MATH_HAS_NATIVE_SIMD
#   undef float2
#   undef float3
#   undef float4
#   undef int2
#   undef int3
#   undef int4
#   endif

    struct float1_storage
    {
        float x;

        MATH_FORCEINLINE float1_storage()
        {
        }

        MATH_FORCEINLINE float1_storage(float x)
            : x(x)
        {
        }

        MATH_FORCEINLINE float1_storage(const float1& v)
        {
            vstore1f(&x, v);
        }

        MATH_FORCEINLINE operator float1() const
        {
            return vload1f(&x);
        }

        MATH_FORCEINLINE float1_storage& operator=(const float1& v)
        {
            vstore1f(&x, v);
            return *this;
        }
    };

    struct float2_storage
    {
        float x, y;

        MATH_FORCEINLINE float2_storage()
        {
        }

        MATH_FORCEINLINE float2_storage(float x)
            : x(x)
            , y(x)
        {
        }

        MATH_FORCEINLINE float2_storage(float x, float y)
            : x(x)
            , y(y)
        {
        }

        MATH_FORCEINLINE float2_storage(const float2& v)
        {
            vstore2f(&x, v);
        }

        MATH_FORCEINLINE operator float2() const
        {
            return vload2f(&x);
        }

        MATH_FORCEINLINE float2_storage& operator=(const float2& v)
        {
            vstore2f(&x, v);
            return *this;
        }
    };

    struct float3_storage
    {
        float x, y, z;

        MATH_FORCEINLINE float3_storage()
        {
        }

        MATH_FORCEINLINE float3_storage(float x)
            : x(x)
            , y(x)
            , z(x)
        {
        }

        MATH_FORCEINLINE float3_storage(float x, float y, float z)
            : x(x)
            , y(y)
            , z(z)
        {
        }

        MATH_FORCEINLINE float3_storage(const float3& v)
        {
            vstore3f(&x, v);
        }

        MATH_FORCEINLINE operator float3() const
        {
            return vload3f(&x);
        }

        MATH_FORCEINLINE float3_storage& operator=(const float3& v)
        {
            vstore3f(&x, v);
            return *this;
        }
    };

    struct float4_storage
    {
        float x, y, z, w;

        MATH_FORCEINLINE float4_storage()
        {
        }

        MATH_FORCEINLINE float4_storage(float x)
            : x(x)
            , y(x)
            , z(x)
            , w(x)
        {
        }

        MATH_FORCEINLINE float4_storage(float x, float y, float z, float w)
            : x(x)
            , y(y)
            , z(z)
            , w(w)
        {
        }

        MATH_FORCEINLINE float4_storage(const float4& v)
        {
            vstore4f(&x, v);
        }

        MATH_FORCEINLINE operator float4() const
        {
            return vload4f(&x);
        }

        MATH_FORCEINLINE float4_storage& operator=(const float4& v)
        {
            vstore4f(&x, v);
            return *this;
        }
    };

    struct alignas(16) float4_storage_aligned
    {
        float4 vec;

        MATH_FORCEINLINE float4_storage_aligned()
        {
        }

        MATH_FORCEINLINE float4_storage_aligned(float x) :
#ifdef MATH_HAS_NATIVE_SIMD
            vec(float4_ctor(x))
#else
            vec(x)
#endif
        {
        }
        MATH_FORCEINLINE float4_storage_aligned(float x, float y, float z, float w) :
#ifdef MATH_HAS_NATIVE_SIMD
            vec(float4_ctor(x, y, z, w))
#else
            vec(x, y, z, w)
#endif
        {
        }
        MATH_FORCEINLINE float4_storage_aligned(const float4& v)
        {
            vec = v;
        }

        MATH_FORCEINLINE float4_storage_aligned(const float4_storage_aligned& v)
        {
            vec = v.vec;
        }

        MATH_FORCEINLINE float4_storage_aligned& operator=(const float4_storage_aligned& v)
        {
            vec = v.vec;
            return *this;
        }

        MATH_FORCEINLINE operator float4() const
        {
            return vec;
        }

        MATH_FORCEINLINE float4_storage_aligned& operator=(const float4& v)
        {
            vec = v;
            return *this;
        }
    };

    struct int1_storage
    {
        int x;

        MATH_FORCEINLINE int1_storage()
        {}
        MATH_FORCEINLINE int1_storage(int x)
            : x(x)
        {}
        MATH_FORCEINLINE int1_storage(const int1& v)
        {
            vstore1i(&x, v);
        }

        MATH_FORCEINLINE operator int1() const
        {
            return vload1i(&x);
        }

        MATH_FORCEINLINE int1_storage& operator=(const int1& v)
        {
            vstore1i(&x, v);
            return *this;
        }
    };

    struct int2_storage
    {
        int x, y;

        MATH_FORCEINLINE int2_storage()
        {}
        MATH_FORCEINLINE int2_storage(int x)
            : x(x)
            , y(x)
        {}
        MATH_FORCEINLINE int2_storage(int x, int y)
            : x(x)
            , y(y)
        {}
        MATH_FORCEINLINE int2_storage(const int2& v)
        {
            vstore2i(&x, v);
        }

        MATH_FORCEINLINE operator int2() const
        {
            return vload2i(&x);
        }

        MATH_FORCEINLINE int2_storage& operator=(const int2& v)
        {
            vstore2i(&x, v);
            return *this;
        }

        MATH_FORCEINLINE bool operator==(const int2_storage& v) const
        {
            return x == v.x && y == v.y;
        }
    };

    struct int3_storage
    {
        int x, y, z;

        MATH_FORCEINLINE int3_storage()
        {}
        MATH_FORCEINLINE int3_storage(int x)
            : x(x)
            , y(x)
            , z(x)
        {}
        MATH_FORCEINLINE int3_storage(int x, int y, int z)
            : x(x)
            , y(y)
            , z(z)
        {}
        MATH_FORCEINLINE int3_storage(const int3& v)
        {
            vstore3i(&x, v);
        }

        MATH_FORCEINLINE operator int3() const
        {
            return vload3i(&x);
        }

        MATH_FORCEINLINE int3_storage& operator=(const int3& v)
        {
            vstore3i(&x, v);
            return *this;
        }

        MATH_FORCEINLINE bool operator==(const int3_storage& v) const
        {
            return x == v.x && y == v.y && z == v.z;
        }
    };

    struct int4_storage
    {
        int x, y, z, w;

        MATH_FORCEINLINE int4_storage()
        {}
        MATH_FORCEINLINE int4_storage(int x)
            : x(x)
            , y(x)
            , z(x)
            , w(x)
        {}
        MATH_FORCEINLINE int4_storage(int x, int y, int z, int w)
            : x(x)
            , y(y)
            , z(z)
            , w(w)
        {}
        MATH_FORCEINLINE int4_storage(const int4& v)
        {
            vstore4i(&x, v);
        }

        MATH_FORCEINLINE operator int4() const
        {
            return vload4i(&x);
        }

        MATH_FORCEINLINE int4_storage& operator=(const int4& v)
        {
            vstore4i(&x, v);
            return *this;
        }
    };

#   ifdef MATH_HAS_NATIVE_SIMD
#   define float3(...) float3_ctor(__VA_ARGS__)
#   define float2(...) float2_ctor(__VA_ARGS__)
#   define float4(...) float4_ctor(__VA_ARGS__)
#   define int3(...) int3_ctor(__VA_ARGS__)
#   define int2(...) int2_ctor(__VA_ARGS__)
#   define int4(...) int4_ctor(__VA_ARGS__)
#   endif

    struct float3x3
    {
        float3 m0, m1, m2;

        MATH_EMPTYINLINE float3x3() {}

        MATH_FORCEINLINE float3x3(const float3x3 &a)
            :   m0(a.m0)
            ,   m1(a.m1)
            ,   m2(a.m2)
        {
        }

        MATH_FORCEINLINE float3x3(const float3 &a, const float3 &b, const float3 &c)
            :   m0(a)
            ,   m1(b)
            ,   m2(c)
        {
        }

        MATH_FORCEINLINE const float3 &operator[](unsigned i) const
        {
            return ((const float3*)&m0)[i];
        }

        MATH_FORCEINLINE float3 &operator[](unsigned i)
        {
            return ((float3*)&m0)[i];
        }

        MATH_FORCEINLINE float3x3 &operator=(const float3x3 &m)
        {
            m0 = m.m0; m1 = m.m1; m2 = m.m2;
            return *this;
        }
    };

    struct float4x4
    {
        float4 m0, m1, m2, m3;

        MATH_EMPTYINLINE float4x4() {}

        MATH_FORCEINLINE float4x4(const float4x4 &a)
            :   m0(a.m0)
            ,   m1(a.m1)
            ,   m2(a.m2)
            ,   m3(a.m3)
        {
        }

        MATH_FORCEINLINE float4x4(const float4 &a, const float4 &b, const float4 &c, const float4 &d)
            :   m0(a)
            ,   m1(b)
            ,   m2(c)
            ,   m3(d)
        {
        }

        MATH_FORCEINLINE const float4 &operator[](unsigned i) const
        {
            return ((const float4*)&m0)[i];
        }

        MATH_FORCEINLINE float4 &operator[](unsigned i)
        {
            return ((float4*)&m0)[i];
        }

        MATH_FORCEINLINE float4x4 &operator=(const float4x4 &m)
        {
            m0 = m.m0; m1 = m.m1; m2 = m.m2; m3 = m.m3;
            return *this;
        }
    };
}

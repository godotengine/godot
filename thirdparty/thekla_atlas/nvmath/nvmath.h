// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MATH_H
#define NV_MATH_H

#include "nvcore/nvcore.h"
#include "nvcore/Debug.h"   // nvDebugCheck
#include "nvcore/Utils.h"   // max, clamp

#include <math.h>

#if NV_OS_WIN32 || NV_OS_XBOX || NV_OS_DURANGO
#include <float.h>  // finite, isnan
#endif

// -- GODOT start --
//#if NV_CPU_X86 || NV_CPU_X86_64
//    //#include <intrin.h>
//    #include <xmmintrin.h>
//#endif
// -- GODOT end --



// Function linkage
#if NVMATH_SHARED
#ifdef NVMATH_EXPORTS
#define NVMATH_API DLL_EXPORT
#define NVMATH_CLASS DLL_EXPORT_CLASS
#else
#define NVMATH_API DLL_IMPORT
#define NVMATH_CLASS DLL_IMPORT
#endif
#else // NVMATH_SHARED
#define NVMATH_API
#define NVMATH_CLASS
#endif // NVMATH_SHARED

// Set some reasonable defaults.
#ifndef NV_USE_ALTIVEC
#   define NV_USE_ALTIVEC NV_CPU_PPC
//#   define NV_USE_ALTIVEC defined(__VEC__)
#endif

#ifndef NV_USE_SSE
#   if NV_CPU_X86_64
        // x64 always supports at least SSE2
#       define NV_USE_SSE 2
#   elif NV_CC_MSVC && defined(_M_IX86_FP)
        // Also on x86 with the /arch:SSE flag in MSVC.
#       define NV_USE_SSE _M_IX86_FP       // 1=SSE, 2=SS2
#   elif defined(__SSE__)
#       define NV_USE_SSE 1
#   elif defined(__SSE2__)
#       define NV_USE_SSE 2
#   else
        // Otherwise we assume no SSE.
#       define NV_USE_SSE 0
#   endif
#endif


// Internally set NV_USE_SIMD when either altivec or sse is available.
#if NV_USE_ALTIVEC && NV_USE_SSE
#	error "Cannot enable both altivec and sse!"
#endif


// -- GODOT start --
#if NV_USE_SSE
    //#include <intrin.h>
    #include <xmmintrin.h>
#endif
// -- GODOT end --


#ifndef PI
#define PI                  float(3.1415926535897932384626433833)
#endif

#define NV_EPSILON          (0.0001f)
#define NV_NORMAL_EPSILON   (0.001f)

/*
#define SQ(r)               ((r)*(r))

#define SIGN_BITMASK        0x80000000

/// Integer representation of a floating-point value.
#define IR(x)               ((uint32 &)(x))

/// Absolute integer representation of a floating-point value
#define AIR(x)              (IR(x) & 0x7fffffff)

/// Floating-point representation of an integer value.
#define FR(x)               ((float&)(x))

/// Integer-based comparison of a floating point value.
/// Don't use it blindly, it can be faster or slower than the FPU comparison, depends on the context.
#define IS_NEGATIVE_FLOAT(x) (IR(x)&SIGN_BITMASK)
*/

extern "C" inline double sqrt_assert(const double f)
{
    nvDebugCheck(f >= 0.0f);
    return sqrt(f);
}

inline float sqrtf_assert(const float f)
{
    nvDebugCheck(f >= 0.0f);
    return sqrtf(f);
}

extern "C" inline double acos_assert(const double f) 
{
    nvDebugCheck(f >= -1.0f && f <= 1.0f);
    return acos(f);
}

inline float acosf_assert(const float f)
{
    nvDebugCheck(f >= -1.0f && f <= 1.0f);
    return acosf(f);
}

extern "C" inline double asin_assert(const double f)
{
    nvDebugCheck(f >= -1.0f && f <= 1.0f);
    return asin(f);
}

inline float asinf_assert(const float f)
{
    nvDebugCheck(f >= -1.0f && f <= 1.0f);
    return asinf(f);
}

// Replace default functions with asserting ones.
#if !NV_CC_MSVC || (NV_CC_MSVC && (_MSC_VER < 1700))    // IC: Apparently this was causing problems in Visual Studio 2012. See Issue 194: https://code.google.com/p/nvidia-texture-tools/issues/detail?id=194
#define sqrt sqrt_assert
#define sqrtf sqrtf_assert
#define acos acos_assert
#define acosf acosf_assert
#define asin asin_assert
#define asinf asinf_assert
#endif

#if NV_CC_MSVC
NV_FORCEINLINE float log2f(float x)
{
    nvCheck(x >= 0);
    return logf(x) / logf(2.0f);
}
NV_FORCEINLINE float exp2f(float x)
{
    return powf(2.0f, x);
}
#endif

namespace nv
{
    inline float toRadian(float degree) { return degree * (PI / 180.0f); }
    inline float toDegree(float radian) { return radian * (180.0f / PI); }

    // Robust floating point comparisons:
    // http://realtimecollisiondetection.net/blog/?p=89
    inline bool equal(const float f0, const float f1, const float epsilon = NV_EPSILON)
    {
        //return fabs(f0-f1) <= epsilon;
        return fabs(f0-f1) <= epsilon * max3(1.0f, fabsf(f0), fabsf(f1));
    }

    inline bool isZero(const float f, const float epsilon = NV_EPSILON)
    {
        return fabs(f) <= epsilon;
    }

    inline bool isFinite(const float f)
    {
#if NV_OS_WIN32 || NV_OS_XBOX || NV_OS_DURANGO
        return _finite(f) != 0;
#elif NV_OS_DARWIN || NV_OS_FREEBSD || NV_OS_OPENBSD || NV_OS_ORBIS
        return isfinite(f);
#elif NV_OS_LINUX
        return finitef(f);
#else
#   error "isFinite not supported"
#endif
        //return std::isfinite (f);
        //return finite (f);
    }

    inline bool isNan(const float f)
    {
#if NV_OS_WIN32 || NV_OS_XBOX || NV_OS_DURANGO
        return _isnan(f) != 0;
#elif NV_OS_DARWIN || NV_OS_FREEBSD || NV_OS_OPENBSD || NV_OS_ORBIS
        return isnan(f);
#elif NV_OS_LINUX
        return isnanf(f);
#else
#   error "isNan not supported"
#endif
    }

    inline uint log2(uint32 i)
    {
        uint32 value = 0;
        while( i >>= 1 ) value++;
        return value;
    }

    inline uint log2(uint64 i)
    {
        uint64 value = 0;
        while (i >>= 1) value++;
        return U32(value);
    }

    inline float lerp(float f0, float f1, float t)
    {
        const float s = 1.0f - t;
        return f0 * s + f1 * t;
    }

    inline float square(float f) { return f * f; }
    inline int square(int i) { return i * i; }

    inline float cube(float f) { return f * f * f; }
    inline int cube(int i) { return i * i * i; }

    inline float frac(float f)
    {
        return f - floor(f);
    }

    inline float floatRound(float f)
    {
        return floorf(f + 0.5f);
    }

    // Eliminates negative zeros from a float array.
    inline void floatCleanup(float * fp, int n)
    {
        for (int i = 0; i < n; i++) {
            //nvDebugCheck(isFinite(fp[i]));
            union { float f; uint32 i; } x = { fp[i] };
            if (x.i == 0x80000000) fp[i] = 0.0f;
        }
    }

    inline float saturate(float f) {
        return clamp(f, 0.0f, 1.0f);
    }

    inline float linearstep(float edge0, float edge1, float x) {
        // Scale, bias and saturate x to 0..1 range
        return saturate((x - edge0) / (edge1 - edge0));
    }

    inline float smoothstep(float edge0, float edge1, float x) {
        x = linearstep(edge0, edge1, x); 

        // Evaluate polynomial
        return x*x*(3 - 2*x);
    }

    inline int sign(float a)
    {
        return (a > 0) - (a < 0);
        //if (a > 0.0f) return 1;
        //if (a < 0.0f) return -1;
        //return 0;
    }

    union Float754 {
        unsigned int raw;
        float value;
        struct {
        #if NV_BIG_ENDIAN
            unsigned int negative:1;
            unsigned int biasedexponent:8;
            unsigned int mantissa:23;
        #else
            unsigned int mantissa:23;
            unsigned int biasedexponent:8;
            unsigned int negative:1;
        #endif
        } field;
    };

    // Return the exponent of x ~ Floor(Log2(x))
    inline int floatExponent(float x)
    {
        Float754 f;
        f.value = x;
        return (f.field.biasedexponent - 127);
    }


    // FloatRGB9E5
    union Float3SE {
        uint32 v;
        struct {
        #if NV_BIG_ENDIAN
            uint32 e : 5;
            uint32 zm : 9;
            uint32 ym : 9;
            uint32 xm : 9;
        #else
            uint32 xm : 9;
            uint32 ym : 9;
            uint32 zm : 9;
            uint32 e : 5;
        #endif
        };
    };

    // FloatR11G11B10
    union Float3PK {
        uint32 v;
        struct {
        #if NV_BIG_ENDIAN
            uint32 ze : 5;
            uint32 zm : 5;
            uint32 ye : 5;
            uint32 ym : 6;
            uint32 xe : 5;
            uint32 xm : 6;
        #else
            uint32 xm : 6;
            uint32 xe : 5;
            uint32 ym : 6;
            uint32 ye : 5;
            uint32 zm : 5;
            uint32 ze : 5;
        #endif
        };
    };


} // nv

#endif // NV_MATH_H

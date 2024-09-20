#pragma once

#include "config.h"

namespace math
{
    static MATH_FORCEINLINE int as_int(const float &x)
    {
        return ((const int*)&x)[0];
    }

    static MATH_FORCEINLINE float as_float(const int &x)
    {
        return ((const float*)&x)[0];
    }

    static MATH_FORCEINLINE int convert_int(float v)
    {
        return (int)v;
    }

    static MATH_FORCEINLINE float convert_float(int v)
    {
        return (float)v;
    }

    static MATH_FORCEINLINE int abs(int x)
    {
#if defined(__GNUC__)
        return __builtin_abs(x);
#else
        return x < 0 ? -x : x;
#endif
    }

    static MATH_FORCEINLINE float abs(float x)
    {
#if defined(__GNUC__)
        return __builtin_fabsf(x);
#else
        return ::fabsf(x);
#endif
    }

    static MATH_FORCEINLINE int min(int x, int y)
    {
        return x < y ? x : y;
    }

    static MATH_FORCEINLINE int max(int x, int y)
    {
        return x > y ? x : y;
    }

//
//  min: minimum
//  return value: minimum value of x and y
//  note: the return value for NaNs and +/- zero are left to the implementation
//
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float min(float x, float y)
    {
        return x < y ? x : y;
    }

#else

    static MATH_FORCEINLINE float min(float x, float y)
    {
#   if defined(_MSC_VER)
        return __min(x, y);
#   elif defined(__ARMCC_VERSION)
        return __builtin_fminf(x, y);
#   else
        return ::fminf(x, y);
#   endif
    }

#endif

//
//  max: maximum
//  return value: maximum value of x and y
//  note: the return value for NaNs and +/- zero are left to the implementation
//
#if defined(MATH_HAS_SIMD_FLOAT)

    static MATH_FORCEINLINE float max(float x, float y)
    {
        return x > y ? x : y;
    }

#else

    static MATH_FORCEINLINE float max(float x, float y)
    {
#   if defined(_MSC_VER)
        return __max(x, y);
#   elif defined(__ARMCC_VERSION)
        return __builtin_fmaxf(x, y);
#   else
        return ::fmaxf(x, y);
#   endif
    }

#endif
}

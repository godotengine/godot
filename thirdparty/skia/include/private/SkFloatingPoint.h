/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkFloatingPoint_DEFINED
#define SkFloatingPoint_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkFloatBits.h"
#include "include/private/SkSafe_math.h"
#include <float.h>
#include <math.h>
#include <cmath>
#include <cstring>
#include <limits>

constexpr float SK_FloatSqrt2 = 1.41421356f;
constexpr float SK_FloatPI    = 3.14159265f;
constexpr double SK_DoublePI  = 3.14159265358979323846264338327950288;

// C++98 cmath std::pow seems to be the earliest portable way to get float pow.
// However, on Linux including cmath undefines isfinite.
// http://gcc.gnu.org/bugzilla/show_bug.cgi?id=14608
static inline float sk_float_pow(float base, float exp) {
    return powf(base, exp);
}

#define sk_float_sqrt(x)        sqrtf(x)
#define sk_float_sin(x)         sinf(x)
#define sk_float_cos(x)         cosf(x)
#define sk_float_tan(x)         tanf(x)
#define sk_float_floor(x)       floorf(x)
#define sk_float_ceil(x)        ceilf(x)
#define sk_float_trunc(x)       truncf(x)
#ifdef SK_BUILD_FOR_MAC
#    define sk_float_acos(x)    static_cast<float>(acos(x))
#    define sk_float_asin(x)    static_cast<float>(asin(x))
#else
#    define sk_float_acos(x)    acosf(x)
#    define sk_float_asin(x)    asinf(x)
#endif
#define sk_float_atan2(y,x)     atan2f(y,x)
#define sk_float_abs(x)         fabsf(x)
#define sk_float_copysign(x, y) copysignf(x, y)
#define sk_float_mod(x,y)       fmodf(x,y)
#define sk_float_exp(x)         expf(x)
#define sk_float_log(x)         logf(x)

constexpr float sk_float_degrees_to_radians(float degrees) {
    return degrees * (SK_FloatPI / 180);
}

constexpr float sk_float_radians_to_degrees(float radians) {
    return radians * (180 / SK_FloatPI);
}

#define sk_float_round(x) sk_float_floor((x) + 0.5f)

// can't find log2f on android, but maybe that just a tool bug?
#ifdef SK_BUILD_FOR_ANDROID
    static inline float sk_float_log2(float x) {
        const double inv_ln_2 = 1.44269504088896;
        return (float)(log(x) * inv_ln_2);
    }
#else
    #define sk_float_log2(x)        log2f(x)
#endif

static inline bool sk_float_isfinite(float x) {
    return SkFloatBits_IsFinite(SkFloat2Bits(x));
}

static inline bool sk_floats_are_finite(float a, float b) {
    return sk_float_isfinite(a) && sk_float_isfinite(b);
}

static inline bool sk_floats_are_finite(const float array[], int count) {
    float prod = 0;
    for (int i = 0; i < count; ++i) {
        prod *= array[i];
    }
    // At this point, prod will either be NaN or 0
    return prod == 0;   // if prod is NaN, this check will return false
}

static inline bool sk_float_isinf(float x) {
    return SkFloatBits_IsInf(SkFloat2Bits(x));
}

static inline bool sk_float_isnan(float x) {
    return !(x == x);
}

#define sk_double_isnan(a)          sk_float_isnan(a)

#define SK_MaxS32FitsInFloat    2147483520
#define SK_MinS32FitsInFloat    -SK_MaxS32FitsInFloat

#define SK_MaxS64FitsInFloat    (SK_MaxS64 >> (63-24) << (63-24))   // 0x7fffff8000000000
#define SK_MinS64FitsInFloat    -SK_MaxS64FitsInFloat

/**
 *  Return the closest int for the given float. Returns SK_MaxS32FitsInFloat for NaN.
 */
static inline int sk_float_saturate2int(float x) {
    x = x < SK_MaxS32FitsInFloat ? x : SK_MaxS32FitsInFloat;
    x = x > SK_MinS32FitsInFloat ? x : SK_MinS32FitsInFloat;
    return (int)x;
}

/**
 *  Return the closest int for the given double. Returns SK_MaxS32 for NaN.
 */
static inline int sk_double_saturate2int(double x) {
    x = x < SK_MaxS32 ? x : SK_MaxS32;
    x = x > SK_MinS32 ? x : SK_MinS32;
    return (int)x;
}

/**
 *  Return the closest int64_t for the given float. Returns SK_MaxS64FitsInFloat for NaN.
 */
static inline int64_t sk_float_saturate2int64(float x) {
    x = x < SK_MaxS64FitsInFloat ? x : SK_MaxS64FitsInFloat;
    x = x > SK_MinS64FitsInFloat ? x : SK_MinS64FitsInFloat;
    return (int64_t)x;
}

#define sk_float_floor2int(x)   sk_float_saturate2int(sk_float_floor(x))
#define sk_float_round2int(x)   sk_float_saturate2int(sk_float_floor((x) + 0.5f))
#define sk_float_ceil2int(x)    sk_float_saturate2int(sk_float_ceil(x))

#define sk_float_floor2int_no_saturate(x)   (int)sk_float_floor(x)
#define sk_float_round2int_no_saturate(x)   (int)sk_float_floor((x) + 0.5f)
#define sk_float_ceil2int_no_saturate(x)    (int)sk_float_ceil(x)

#define sk_double_floor(x)          floor(x)
#define sk_double_round(x)          floor((x) + 0.5)
#define sk_double_ceil(x)           ceil(x)
#define sk_double_floor2int(x)      (int)floor(x)
#define sk_double_round2int(x)      (int)floor((x) + 0.5)
#define sk_double_ceil2int(x)       (int)ceil(x)

// Cast double to float, ignoring any warning about too-large finite values being cast to float.
// Clang thinks this is undefined, but it's actually implementation defined to return either
// the largest float or infinity (one of the two bracketing representable floats).  Good enough!
SK_ATTRIBUTE(no_sanitize("float-cast-overflow"))
static inline float sk_double_to_float(double x) {
    return static_cast<float>(x);
}

#define SK_FloatNaN                 std::numeric_limits<float>::quiet_NaN()
#define SK_FloatInfinity            (+std::numeric_limits<float>::infinity())
#define SK_FloatNegativeInfinity    (-std::numeric_limits<float>::infinity())

#define SK_DoubleNaN                std::numeric_limits<double>::quiet_NaN()

// Returns false if any of the floats are outside of [0...1]
// Returns true if count is 0
bool sk_floats_are_unit(const float array[], size_t count);

static inline float sk_float_rsqrt_portable(float x) { return 1.0f / sk_float_sqrt(x); }
static inline float sk_float_rsqrt         (float x) { return 1.0f / sk_float_sqrt(x); }

// Returns the log2 of the provided value, were that value to be rounded up to the next power of 2.
// Returns 0 if value <= 0:
// Never returns a negative number, even if value is NaN.
//
//     sk_float_nextlog2((-inf..1]) -> 0
//     sk_float_nextlog2((1..2]) -> 1
//     sk_float_nextlog2((2..4]) -> 2
//     sk_float_nextlog2((4..8]) -> 3
//     ...
static inline int sk_float_nextlog2(float x) {
    uint32_t bits = (uint32_t)SkFloat2Bits(x);
    bits += (1u << 23) - 1u;  // Increment the exponent for non-powers-of-2.
    int exp = ((int32_t)bits >> 23) - 127;
    return exp & ~(exp >> 31);  // Return 0 for negative or denormalized floats, and exponents < 0.
}

// This is the number of significant digits we can print in a string such that when we read that
// string back we get the floating point number we expect.  The minimum value C requires is 6, but
// most compilers support 9
#ifdef FLT_DECIMAL_DIG
#define SK_FLT_DECIMAL_DIG FLT_DECIMAL_DIG
#else
#define SK_FLT_DECIMAL_DIG 9
#endif

// IEEE defines how float divide behaves for non-finite values and zero-denoms, but C does not
// so we have a helper that suppresses the possible undefined-behavior warnings.

SK_ATTRIBUTE(no_sanitize("float-divide-by-zero"))
static inline float sk_ieee_float_divide(float numer, float denom) {
    return numer / denom;
}

SK_ATTRIBUTE(no_sanitize("float-divide-by-zero"))
static inline double sk_ieee_double_divide(double numer, double denom) {
    return numer / denom;
}

// While we clean up divide by zero, we'll replace places that do divide by zero with this TODO.
static inline float sk_ieee_float_divide_TODO_IS_DIVIDE_BY_ZERO_SAFE_HERE(float n, float d) {
    return sk_ieee_float_divide(n,d);
}

static inline float sk_fmaf(float f, float m, float a) {
#if defined(FP_FAST_FMA)
    return std::fmaf(f,m,a);
#else
    return f*m+a;
#endif
}

#endif

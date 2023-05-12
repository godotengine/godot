// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2021 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/*
 * This module implements a variety of mathematical data types and library
 * functions used by the codec.
 */

#ifndef ASTC_MATHLIB_H_INCLUDED
#define ASTC_MATHLIB_H_INCLUDED

#include <cassert>
#include <cstdint>
#include <cmath>

#ifndef ASTCENC_POPCNT
  #if defined(__POPCNT__)
    #define ASTCENC_POPCNT 1
  #else
    #define ASTCENC_POPCNT 0
  #endif
#endif

#ifndef ASTCENC_F16C
  #if defined(__F16C__)
    #define ASTCENC_F16C 1
  #else
    #define ASTCENC_F16C 0
  #endif
#endif

#ifndef ASTCENC_SSE
  #if defined(__SSE4_2__)
    #define ASTCENC_SSE 42
  #elif defined(__SSE4_1__)
    #define ASTCENC_SSE 41
  #elif defined(__SSE2__)
    #define ASTCENC_SSE 20
  #else
    #define ASTCENC_SSE 0
  #endif
#endif

#ifndef ASTCENC_AVX
  #if defined(__AVX2__)
    #define ASTCENC_AVX 2
  #elif defined(__AVX__)
    #define ASTCENC_AVX 1
  #else
    #define ASTCENC_AVX 0
  #endif
#endif

#ifndef ASTCENC_NEON
  #if defined(__aarch64__)
    #define ASTCENC_NEON 1
  #else
    #define ASTCENC_NEON 0
  #endif
#endif

#if ASTCENC_AVX
  #define ASTCENC_VECALIGN 32
#else
  #define ASTCENC_VECALIGN 16
#endif

#if ASTCENC_SSE != 0 || ASTCENC_AVX != 0 || ASTCENC_POPCNT != 0
	#include <immintrin.h>
#endif

/* ============================================================================
  Fast math library; note that many of the higher-order functions in this set
  use approximations which are less accurate, but faster, than <cmath> standard
  library equivalents.

  Note: Many of these are not necessarily faster than simple C versions when
  used on a single scalar value, but are included for testing purposes as most
  have an option based on SSE intrinsics and therefore provide an obvious route
  to future vectorization.
============================================================================ */

// Union for manipulation of float bit patterns
typedef union
{
	uint32_t u;
	int32_t s;
	float f;
} if32;

// These are namespaced to avoid colliding with C standard library functions.
namespace astc
{

static const float PI          = 3.14159265358979323846f;
static const float PI_OVER_TWO = 1.57079632679489661923f;

/**
 * @brief SP float absolute value.
 *
 * @param v   The value to make absolute.
 *
 * @return The absolute value.
 */
static inline float fabs(float v)
{
	return std::fabs(v);
}

/**
 * @brief Test if a float value is a nan.
 *
 * @param v    The value test.
 *
 * @return Zero is not a NaN, non-zero otherwise.
 */
static inline bool isnan(float v)
{
	return v != v;
}

/**
 * @brief Return the minimum of two values.
 *
 * For floats, NaNs are turned into @c q.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 *
 * @return The smallest value.
 */
template<typename T>
static inline T min(T p, T q)
{
	return p < q ? p : q;
}

/**
 * @brief Return the minimum of three values.
 *
 * For floats, NaNs are turned into @c r.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 * @param r   The third value to compare.
 *
 * @return The smallest value.
 */
template<typename T>
static inline T min(T p, T q, T r)
{
	return min(min(p, q), r);
}

/**
 * @brief Return the minimum of four values.
 *
 * For floats, NaNs are turned into @c s.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 * @param r   The third value to compare.
 * @param s   The fourth value to compare.
 *
 * @return The smallest value.
 */
template<typename T>
static inline T min(T p, T q, T r, T s)
{
	return min(min(p, q), min(r, s));
}

/**
 * @brief Return the maximum of two values.
 *
 * For floats, NaNs are turned into @c q.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 *
 * @return The largest value.
 */
template<typename T>
static inline T max(T p, T q)
{
	return p > q ? p : q;
}

/**
 * @brief Return the maximum of three values.
 *
 * For floats, NaNs are turned into @c r.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 * @param r   The third value to compare.
 *
 * @return The largest value.
 */
template<typename T>
static inline T max(T p, T q, T r)
{
	return max(max(p, q), r);
}

/**
 * @brief Return the maximum of four values.
 *
 * For floats, NaNs are turned into @c s.
 *
 * @param p   The first value to compare.
 * @param q   The second value to compare.
 * @param r   The third value to compare.
 * @param s   The fourth value to compare.
 *
 * @return The largest value.
 */
template<typename T>
static inline T max(T p, T q, T r, T s)
{
	return max(max(p, q), max(r, s));
}

/**
 * @brief Clamp a value value between @c mn and @c mx.
 *
 * For floats, NaNs are turned into @c mn.
 *
 * @param v      The value to clamp.
 * @param mn     The min value (inclusive).
 * @param mx     The max value (inclusive).
 *
 * @return The clamped value.
 */
template<typename T>
inline T clamp(T v, T mn, T mx)
{
	// Do not reorder; correct NaN handling relies on the fact that comparison
	// with NaN returns false and will fall-though to the "min" value.
	if (v > mx) return mx;
	if (v > mn) return v;
	return mn;
}

/**
 * @brief Clamp a float value between 0.0f and 1.0f.
 *
 * NaNs are turned into 0.0f.
 *
 * @param v   The value to clamp.
 *
 * @return The clamped value.
 */
static inline float clamp1f(float v)
{
	return astc::clamp(v, 0.0f, 1.0f);
}

/**
 * @brief Clamp a float value between 0.0f and 255.0f.
 *
 * NaNs are turned into 0.0f.
 *
 * @param v  The value to clamp.
 *
 * @return The clamped value.
 */
static inline float clamp255f(float v)
{
	return astc::clamp(v, 0.0f, 255.0f);
}

/**
 * @brief SP float round-down.
 *
 * @param v   The value to round.
 *
 * @return The rounded value.
 */
static inline float flt_rd(float v)
{
	return std::floor(v);
}

/**
 * @brief SP float round-to-nearest and convert to integer.
 *
 * @param v   The value to round.
 *
 * @return The rounded value.
 */
static inline int flt2int_rtn(float v)
{

	return static_cast<int>(v + 0.5f);
}

/**
 * @brief SP float round down and convert to integer.
 *
 * @param v   The value to round.
 *
 * @return The rounded value.
 */
static inline int flt2int_rd(float v)
{
	return static_cast<int>(v);
}

/**
 * @brief SP float bit-interpreted as an integer.
 *
 * @param v   The value to bitcast.
 *
 * @return The converted value.
 */
static inline int float_as_int(float v)
{
	union { int a; float b; } u;
	u.b = v;
	return u.a;
}

/**
 * @brief Integer bit-interpreted as an SP float.
 *
 * @param v   The value to bitcast.
 *
 * @return The converted value.
 */
static inline float int_as_float(int v)
{
	union { int a; float b; } u;
	u.a = v;
	return u.b;
}

/**
 * @brief Fast approximation of 1.0 / sqrt(val).
 *
 * @param v   The input value.
 *
 * @return The approximated result.
 */
static inline float rsqrt(float v)
{
	return 1.0f / std::sqrt(v);
}

/**
 * @brief Fast approximation of sqrt(val).
 *
 * @param v   The input value.
 *
 * @return The approximated result.
 */
static inline float sqrt(float v)
{
	return std::sqrt(v);
}

/**
 * @brief Extract mantissa and exponent of a float value.
 *
 * @param      v      The input value.
 * @param[out] expo   The output exponent.
 *
 * @return The mantissa.
 */
static inline float frexp(float v, int* expo)
{
	if32 p;
	p.f = v;
	*expo = ((p.u >> 23) & 0xFF) - 126;
	p.u = (p.u & 0x807fffff) | 0x3f000000;
	return p.f;
}

/**
 * @brief Initialize the seed structure for a random number generator.
 *
 * Important note: For the purposes of ASTC we want sets of random numbers to
 * use the codec, but we want the same seed value across instances and threads
 * to ensure that image output is stable across compressor runs and across
 * platforms. Every PRNG created by this call will therefore return the same
 * sequence of values ...
 *
 * @param state The state structure to initialize.
 */
void rand_init(uint64_t state[2]);

/**
 * @brief Return the next random number from the generator.
 *
 * This RNG is an implementation of the "xoroshoro-128+ 1.0" PRNG, based on the
 * public-domain implementation given by David Blackman & Sebastiano Vigna at
 * http://vigna.di.unimi.it/xorshift/xoroshiro128plus.c
 *
 * @param state The state structure to use/update.
 */
uint64_t rand(uint64_t state[2]);

}

/* ============================================================================
  Softfloat library with fp32 and fp16 conversion functionality.
============================================================================ */
#if (ASTCENC_F16C == 0) && (ASTCENC_NEON == 0)
	/* narrowing float->float conversions */
	uint16_t float_to_sf16(float val);
	float sf16_to_float(uint16_t val);
#endif

/*********************************
  Vector library
*********************************/
#include "astcenc_vecmathlib.h"

/*********************************
  Declaration of line types
*********************************/
// parametric line, 2D: The line is given by line = a + b * t.

struct line2
{
	vfloat4 a;
	vfloat4 b;
};

// parametric line, 3D
struct line3
{
	vfloat4 a;
	vfloat4 b;
};

struct line4
{
	vfloat4 a;
	vfloat4 b;
};


struct processed_line2
{
	vfloat4 amod;
	vfloat4 bs;
};

struct processed_line3
{
	vfloat4 amod;
	vfloat4 bs;
};

struct processed_line4
{
	vfloat4 amod;
	vfloat4 bs;
};

#endif

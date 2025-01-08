// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2019-2022 Arm Limited
// Copyright 2008 Jose Fonseca
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
 * This module implements vector support for floats, ints, and vector lane
 * control masks. It provides access to both explicit vector width types, and
 * flexible N-wide types where N can be determined at compile time.
 *
 * The design of this module encourages use of vector length agnostic code, via
 * the vint, vfloat, and vmask types. These will take on the widest SIMD vector
 * with that is available at compile time. The current vector width is
 * accessible for e.g. loop strides via the ASTCENC_SIMD_WIDTH constant.
 *
 * Explicit scalar types are accessible via the vint1, vfloat1, vmask1 types.
 * These are provided primarily for prototyping and algorithm debug of VLA
 * implementations.
 *
 * Explicit 4-wide types are accessible via the vint4, vfloat4, and vmask4
 * types. These are provided for use by VLA code, but are also expected to be
 * used as a fixed-width type and will supported a reference C++ fallback for
 * use on platforms without SIMD intrinsics.
 *
 * Explicit 8-wide types are accessible via the vint8, vfloat8, and vmask8
 * types. These are provide for use by VLA code, and are not expected to be
 * used as a fixed-width type in normal code. No reference C implementation is
 * provided on platforms without underlying SIMD intrinsics.
 *
 * With the current implementation ISA support is provided for:
 *
 *     * 1-wide for scalar reference.
 *     * 4-wide for Armv8-A NEON.
 *     * 4-wide for x86-64 SSE2.
 *     * 4-wide for x86-64 SSE4.1.
 *     * 8-wide for x86-64 AVX2.
 */

#ifndef ASTC_VECMATHLIB_H_INCLUDED
#define ASTC_VECMATHLIB_H_INCLUDED

#if ASTCENC_SSE != 0 || ASTCENC_AVX != 0
	#include <immintrin.h>
#elif ASTCENC_NEON != 0
	#include <arm_neon.h>
#endif

#if !defined(__clang__) && defined(_MSC_VER)
	#define ASTCENC_SIMD_INLINE __forceinline
	#define ASTCENC_NO_INLINE
#elif defined(__GNUC__) && !defined(__clang__)
	#define ASTCENC_SIMD_INLINE __attribute__((always_inline)) inline
	#define ASTCENC_NO_INLINE __attribute__ ((noinline))
#else
	#define ASTCENC_SIMD_INLINE __attribute__((always_inline, nodebug)) inline
	#define ASTCENC_NO_INLINE __attribute__ ((noinline))
#endif

#if ASTCENC_AVX >= 2
	/* If we have AVX2 expose 8-wide VLA. */
	#include "astcenc_vecmathlib_sse_4.h"
	#include "astcenc_vecmathlib_common_4.h"
	#include "astcenc_vecmathlib_avx2_8.h"

	#define ASTCENC_SIMD_WIDTH 8

	using vfloat = vfloat8;

	#if defined(ASTCENC_NO_INVARIANCE)
		using vfloatacc = vfloat8;
	#else
		using vfloatacc = vfloat4;
	#endif

	using vint = vint8;
	using vmask = vmask8;

	constexpr auto loada = vfloat8::loada;
	constexpr auto load1 = vfloat8::load1;

#elif ASTCENC_SSE >= 20
	/* If we have SSE expose 4-wide VLA, and 4-wide fixed width. */
	#include "astcenc_vecmathlib_sse_4.h"
	#include "astcenc_vecmathlib_common_4.h"

	#define ASTCENC_SIMD_WIDTH 4

	using vfloat = vfloat4;
	using vfloatacc = vfloat4;
	using vint = vint4;
	using vmask = vmask4;

	constexpr auto loada = vfloat4::loada;
	constexpr auto load1 = vfloat4::load1;

#elif ASTCENC_NEON > 0
	/* If we have NEON expose 4-wide VLA. */
	#include "astcenc_vecmathlib_neon_4.h"
	#include "astcenc_vecmathlib_common_4.h"

	#define ASTCENC_SIMD_WIDTH 4

	using vfloat = vfloat4;
	using vfloatacc = vfloat4;
	using vint = vint4;
	using vmask = vmask4;

	constexpr auto loada = vfloat4::loada;
	constexpr auto load1 = vfloat4::load1;

#else
	// If we have nothing expose 4-wide VLA, and 4-wide fixed width.

	// Note: We no longer expose the 1-wide scalar fallback because it is not
	// invariant with the 4-wide path due to algorithms that use horizontal
	// operations that accumulate a local vector sum before accumulating into
	// a running sum.
	//
	// For 4 items adding into an accumulator using 1-wide vectors the sum is:
	//
	//     result = ((((sum + l0) + l1) + l2) + l3)
	//
    // ... whereas the accumulator for a 4-wide vector sum is:
	//
	//     result = sum + ((l0 + l2) + (l1 + l3))
	//
	// In "normal maths" this is the same, but the floating point reassociation
	// differences mean that these will not produce the same result.

	#include "astcenc_vecmathlib_none_4.h"
	#include "astcenc_vecmathlib_common_4.h"

	#define ASTCENC_SIMD_WIDTH 4

	using vfloat = vfloat4;
	using vfloatacc = vfloat4;
	using vint = vint4;
	using vmask = vmask4;

	constexpr auto loada = vfloat4::loada;
	constexpr auto load1 = vfloat4::load1;
#endif

/**
 * @brief Round a count down to the largest multiple of 8.
 *
 * @param count   The unrounded value.
 *
 * @return The rounded value.
 */
ASTCENC_SIMD_INLINE unsigned int round_down_to_simd_multiple_8(unsigned int count)
{
	return count & static_cast<unsigned int>(~(8 - 1));
}

/**
 * @brief Round a count down to the largest multiple of 4.
 *
 * @param count   The unrounded value.
 *
 * @return The rounded value.
 */
ASTCENC_SIMD_INLINE unsigned int round_down_to_simd_multiple_4(unsigned int count)
{
	return count & static_cast<unsigned int>(~(4 - 1));
}

/**
 * @brief Round a count down to the largest multiple of the SIMD width.
 *
 * Assumption that the vector width is a power of two ...
 *
 * @param count   The unrounded value.
 *
 * @return The rounded value.
 */
ASTCENC_SIMD_INLINE unsigned int round_down_to_simd_multiple_vla(unsigned int count)
{
	return count & static_cast<unsigned int>(~(ASTCENC_SIMD_WIDTH - 1));
}

/**
 * @brief Round a count up to the largest multiple of the SIMD width.
 *
 * Assumption that the vector width is a power of two ...
 *
 * @param count   The unrounded value.
 *
 * @return The rounded value.
 */
ASTCENC_SIMD_INLINE unsigned int round_up_to_simd_multiple_vla(unsigned int count)
{
	unsigned int multiples = (count + ASTCENC_SIMD_WIDTH - 1) / ASTCENC_SIMD_WIDTH;
	return multiples * ASTCENC_SIMD_WIDTH;
}

/**
 * @brief Return @c a with lanes negated if the @c b lane is negative.
 */
ASTCENC_SIMD_INLINE vfloat change_sign(vfloat a, vfloat b)
{
	vint ia = float_as_int(a);
	vint ib = float_as_int(b);
	vint sign_mask(static_cast<int>(0x80000000));
	vint r = ia ^ (ib & sign_mask);
	return int_as_float(r);
}

/**
 * @brief Return fast, but approximate, vector atan(x).
 *
 * Max error of this implementation is 0.004883.
 */
ASTCENC_SIMD_INLINE vfloat atan(vfloat x)
{
	vmask c = abs(x) > vfloat(1.0f);
	vfloat z = change_sign(vfloat(astc::PI_OVER_TWO), x);
	vfloat y = select(x, vfloat(1.0f) / x, c);
	y = y / (y * y * vfloat(0.28f) + vfloat(1.0f));
	return select(y, z - y, c);
}

/**
 * @brief Return fast, but approximate, vector atan2(x, y).
 */
ASTCENC_SIMD_INLINE vfloat atan2(vfloat y, vfloat x)
{
	vfloat z = atan(abs(y / x));
	vmask xmask = vmask(float_as_int(x).m);
	return change_sign(select_msb(z, vfloat(astc::PI) - z, xmask), y);
}

/*
 * @brief Factory that returns a unit length 4 component vfloat4.
 */
static ASTCENC_SIMD_INLINE vfloat4 unit4()
{
	return vfloat4(0.5f);
}

/**
 * @brief Factory that returns a unit length 3 component vfloat4.
 */
static ASTCENC_SIMD_INLINE vfloat4 unit3()
{
	float val = 0.577350258827209473f;
	return vfloat4(val, val, val, 0.0f);
}

/**
 * @brief Factory that returns a unit length 2 component vfloat4.
 */
static ASTCENC_SIMD_INLINE vfloat4 unit2()
{
	float val = 0.707106769084930420f;
	return vfloat4(val, val, 0.0f, 0.0f);
}

/**
 * @brief Factory that returns a 3 component vfloat4.
 */
static ASTCENC_SIMD_INLINE vfloat4 vfloat3(float a, float b, float c)
{
	return vfloat4(a, b, c, 0.0f);
}

/**
 * @brief Factory that returns a 2 component vfloat4.
 */
static ASTCENC_SIMD_INLINE vfloat4 vfloat2(float a, float b)
{
	return vfloat4(a, b, 0.0f, 0.0f);
}

/**
 * @brief Normalize a non-zero length vector to unit length.
 */
static ASTCENC_SIMD_INLINE vfloat4 normalize(vfloat4 a)
{
	vfloat4 length = dot(a, a);
	return a / sqrt(length);
}

/**
 * @brief Normalize a vector, returning @c safe if len is zero.
 */
static ASTCENC_SIMD_INLINE vfloat4 normalize_safe(vfloat4 a, vfloat4 safe)
{
	vfloat4 length = dot(a, a);
	if (length.lane<0>() != 0.0f)
	{
		return a / sqrt(length);
	}

	return safe;
}



#define POLY0(x, c0)                     (                                     c0)
#define POLY1(x, c0, c1)                 ((POLY0(x, c1) * x)                 + c0)
#define POLY2(x, c0, c1, c2)             ((POLY1(x, c1, c2) * x)             + c0)
#define POLY3(x, c0, c1, c2, c3)         ((POLY2(x, c1, c2, c3) * x)         + c0)
#define POLY4(x, c0, c1, c2, c3, c4)     ((POLY3(x, c1, c2, c3, c4) * x)     + c0)
#define POLY5(x, c0, c1, c2, c3, c4, c5) ((POLY4(x, c1, c2, c3, c4, c5) * x) + c0)

/**
 * @brief Compute an approximate exp2(x) for each lane in the vector.
 *
 * Based on 5th degree minimax polynomials, ported from this blog
 * https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
 */
static ASTCENC_SIMD_INLINE vfloat4 exp2(vfloat4 x)
{
	x = clamp(-126.99999f, 129.0f, x);

	vint4 ipart = float_to_int(x - 0.5f);
	vfloat4 fpart = x - int_to_float(ipart);

	// Integer contrib, using 1 << ipart
	vfloat4 iexp = int_as_float(lsl<23>(ipart + 127));

	// Fractional contrib, using polynomial fit of 2^x in range [-0.5, 0.5)
	vfloat4 fexp = POLY5(fpart,
	                     9.9999994e-1f,
	                     6.9315308e-1f,
	                     2.4015361e-1f,
	                     5.5826318e-2f,
	                     8.9893397e-3f,
	                     1.8775767e-3f);

	return iexp * fexp;
}

/**
 * @brief Compute an approximate log2(x) for each lane in the vector.
 *
 * Based on 5th degree minimax polynomials, ported from this blog
 * https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
 */
static ASTCENC_SIMD_INLINE vfloat4 log2(vfloat4 x)
{
	vint4 exp(0x7F800000);
	vint4 mant(0x007FFFFF);
	vint4 one(0x3F800000);

	vint4 i = float_as_int(x);

	vfloat4 e = int_to_float(lsr<23>(i & exp) - 127);

	vfloat4 m = int_as_float((i & mant) | one);

	// Polynomial fit of log2(x)/(x - 1), for x in range [1, 2)
	vfloat4 p = POLY4(m,
	                  2.8882704548164776201f,
	                 -2.52074962577807006663f,
	                  1.48116647521213171641f,
	                 -0.465725644288844778798f,
	                  0.0596515482674574969533f);

	// Increases the polynomial degree, but ensures that log2(1) == 0
	p = p * (m - 1.0f);

	return p + e;
}

/**
 * @brief Compute an approximate pow(x, y) for each lane in the vector.
 *
 * Power function based on the exp2(log2(x) * y) transform.
 */
static ASTCENC_SIMD_INLINE vfloat4 pow(vfloat4 x, vfloat4 y)
{
	vmask4 zero_mask = y == vfloat4(0.0f);
	vfloat4 estimate = exp2(log2(x) * y);

	// Guarantee that y == 0 returns exactly 1.0f
	return select(estimate, vfloat4(1.0f), zero_mask);
}

/**
 * @brief Count the leading zeros for each lane in @c a.
 *
 * Valid for all data values of @c a; will return a per-lane value [0, 32].
 */
static ASTCENC_SIMD_INLINE vint4 clz(vint4 a)
{
	// This function is a horrible abuse of floating point exponents to convert
	// the original integer value into a 2^N encoding we can recover easily.

	// Convert to float without risk of rounding up by keeping only top 8 bits.
	// This trick is is guaranteed to keep top 8 bits and clear the 9th.
	a = (~lsr<8>(a)) & a;
	a = float_as_int(int_to_float(a));

	// Extract and unbias exponent
	a = vint4(127 + 31) - lsr<23>(a);

	// Clamp result to a valid 32-bit range
	return clamp(0, 32, a);
}

/**
 * @brief Return lanewise 2^a for each lane in @c a.
 *
 * Use of signed int means that this is only valid for values in range [0, 31].
 */
static ASTCENC_SIMD_INLINE vint4 two_to_the_n(vint4 a)
{
	// 2^30 is the largest signed number than can be represented
	assert(all(a < vint4(31)));

	// This function is a horrible abuse of floating point to use the exponent
	// and float conversion to generate a 2^N multiple.

	// Bias the exponent
	vint4 exp = a + 127;
	exp = lsl<23>(exp);

	// Reinterpret the bits as a float, and then convert to an int
	vfloat4 f = int_as_float(exp);
	return float_to_int(f);
}

/**
 * @brief Convert unorm16 [0, 65535] to float16 in range [0, 1].
 */
static ASTCENC_SIMD_INLINE vint4 unorm16_to_sf16(vint4 p)
{
	vint4 fp16_one = vint4(0x3C00);
	vint4 fp16_small = lsl<8>(p);

	vmask4 is_one = p == vint4(0xFFFF);
	vmask4 is_small = p < vint4(4);

	// Manually inline clz() on Visual Studio to avoid release build codegen bug
	// see https://github.com/ARM-software/astc-encoder/issues/259
#if !defined(__clang__) && defined(_MSC_VER)
	vint4 a = (~lsr<8>(p)) & p;
	a = float_as_int(int_to_float(a));
	a = vint4(127 + 31) - lsr<23>(a);
	vint4 lz = clamp(0, 32, a) - 16;
#else
	vint4 lz = clz(p) - 16;
#endif

	p = p * two_to_the_n(lz + 1);
	p = p & vint4(0xFFFF);

	p = lsr<6>(p);

	p = p | lsl<10>(vint4(14) - lz);

	vint4 r = select(p, fp16_one, is_one);
	r = select(r, fp16_small, is_small);
	return r;
}

/**
 * @brief Convert 16-bit LNS to float16.
 */
static ASTCENC_SIMD_INLINE vint4 lns_to_sf16(vint4 p)
{
	vint4 mc = p & 0x7FF;
	vint4 ec = lsr<11>(p);

	vint4 mc_512 = mc * 3;
	vmask4 mask_512 = mc < vint4(512);

	vint4 mc_1536 = mc * 4 - 512;
	vmask4 mask_1536 = mc < vint4(1536);

	vint4 mc_else = mc * 5 - 2048;

	vint4 mt = mc_else;
	mt = select(mt, mc_1536, mask_1536);
	mt = select(mt, mc_512, mask_512);

	vint4 res = lsl<10>(ec) | lsr<3>(mt);
	return min(res, vint4(0x7BFF));
}

/**
 * @brief Extract mantissa and exponent of a float value.
 *
 * @param      a      The input value.
 * @param[out] exp    The output exponent.
 *
 * @return The mantissa.
 */
static ASTCENC_SIMD_INLINE vfloat4 frexp(vfloat4 a, vint4& exp)
{
	// Interpret the bits as an integer
	vint4 ai = float_as_int(a);

	// Extract and unbias the exponent
	exp = (lsr<23>(ai) & 0xFF) - 126;

	// Extract and unbias the mantissa
	vint4 manti = (ai &  static_cast<int>(0x807FFFFF)) | 0x3F000000;
	return int_as_float(manti);
}

/**
 * @brief Convert float to 16-bit LNS.
 */
static ASTCENC_SIMD_INLINE vfloat4 float_to_lns(vfloat4 a)
{
	vint4 exp;
	vfloat4 mant = frexp(a, exp);

	// Do these early before we start messing about ...
	vmask4 mask_underflow_nan = ~(a > vfloat4(1.0f / 67108864.0f));
	vmask4 mask_infinity = a >= vfloat4(65536.0f);

	// If input is smaller than 2^-14, multiply by 2^25 and don't bias.
	vmask4 exp_lt_m13 = exp < vint4(-13);

	vfloat4 a1a = a * 33554432.0f;
	vint4 expa = vint4::zero();

	vfloat4 a1b = (mant - 0.5f) * 4096;
	vint4 expb = exp + 14;

	a = select(a1b, a1a, exp_lt_m13);
	exp = select(expb, expa, exp_lt_m13);

	vmask4 a_lt_384 = a < vfloat4(384.0f);
	vmask4 a_lt_1408 = a <= vfloat4(1408.0f);

	vfloat4 a2a = a * (4.0f / 3.0f);
	vfloat4 a2b = a + 128.0f;
	vfloat4 a2c = (a + 512.0f) * (4.0f / 5.0f);

	a = a2c;
	a = select(a, a2b, a_lt_1408);
	a = select(a, a2a, a_lt_384);

	a = a + (int_to_float(exp) * 2048.0f) + 1.0f;

	a = select(a, vfloat4(65535.0f), mask_infinity);
	a = select(a, vfloat4::zero(), mask_underflow_nan);

	return a;
}

namespace astc
{

static ASTCENC_SIMD_INLINE float pow(float x, float y)
{
	return pow(vfloat4(x), vfloat4(y)).lane<0>();
}

}

#endif // #ifndef ASTC_VECMATHLIB_H_INCLUDED

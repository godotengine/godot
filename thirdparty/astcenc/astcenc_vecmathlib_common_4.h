// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2020-2024 Arm Limited
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

/**
 * @brief Generic 4x32-bit vector functions.
 *
 * This module implements generic 4-wide vector functions that are valid for
 * all instruction sets, typically implemented using lower level 4-wide
 * operations that are ISA-specific.
 */

#ifndef ASTC_VECMATHLIB_COMMON_4_H_INCLUDED
#define ASTC_VECMATHLIB_COMMON_4_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

// ============================================================================
// vmask4 operators and functions
// ============================================================================

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask4 a)
{
	return mask(a) != 0;
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask4 a)
{
	return mask(a) == 0xF;
}

// ============================================================================
// vint4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by scalar addition.
 */
ASTCENC_SIMD_INLINE vint4 operator+(vint4 a, int b)
{
	return a + vint4(b);
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vint4& operator+=(vint4& a, const vint4& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by scalar subtraction.
 */
ASTCENC_SIMD_INLINE vint4 operator-(vint4 a, int b)
{
	return a - vint4(b);
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vint4 operator*(vint4 a, int b)
{
	return a * vint4(b);
}

/**
 * @brief Overload: vector by scalar bitwise or.
 */
ASTCENC_SIMD_INLINE vint4 operator|(vint4 a, int b)
{
	return a | vint4(b);
}

/**
 * @brief Overload: vector by scalar bitwise and.
 */
ASTCENC_SIMD_INLINE vint4 operator&(vint4 a, int b)
{
	return a & vint4(b);
}

/**
 * @brief Overload: vector by scalar bitwise xor.
 */
ASTCENC_SIMD_INLINE vint4 operator^(vint4 a, int b)
{
	return a ^ vint4(b);
}

/**
 * @brief Return the clamped value between min and max.
 */
ASTCENC_SIMD_INLINE vint4 clamp(int minv, int maxv, vint4 a)
{
	return min(max(a, vint4(minv)), vint4(maxv));
}

/**
 * @brief Return the horizontal sum of RGB vector lanes as a scalar.
 */
ASTCENC_SIMD_INLINE int hadd_rgb_s(vint4 a)
{
	return a.lane<0>() + a.lane<1>() + a.lane<2>();
}

// ============================================================================
// vfloat4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vfloat4& operator+=(vfloat4& a, const vfloat4& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by scalar addition.
 */
ASTCENC_SIMD_INLINE vfloat4 operator+(vfloat4 a, float b)
{
	return a + vfloat4(b);
}

/**
 * @brief Overload: vector by scalar subtraction.
 */
ASTCENC_SIMD_INLINE vfloat4 operator-(vfloat4 a, float b)
{
	return a - vfloat4(b);
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vfloat4 operator*(vfloat4 a, float b)
{
	return a * vfloat4(b);
}

/**
 * @brief Overload: scalar by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat4 operator*(float a, vfloat4 b)
{
	return vfloat4(a) * b;
}

/**
 * @brief Overload: vector by scalar division.
 */
ASTCENC_SIMD_INLINE vfloat4 operator/(vfloat4 a, float b)
{
	return a / vfloat4(b);
}

/**
 * @brief Overload: scalar by vector division.
 */
ASTCENC_SIMD_INLINE vfloat4 operator/(float a, vfloat4 b)
{
	return vfloat4(a) / b;
}

/**
 * @brief Return the min vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 min(vfloat4 a, float b)
{
	return min(a, vfloat4(b));
}

/**
 * @brief Return the max vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 max(vfloat4 a, float b)
{
	return max(a, vfloat4(b));
}

/**
 * @brief Return the clamped value between min and max.
 *
 * It is assumed that neither @c min nor @c max are NaN values. If @c a is NaN
 * then @c min will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 clamp(float minv, float maxv, vfloat4 a)
{
	// Do not reorder - second operand will return if either is NaN
	return min(max(a, minv), maxv);
}

/**
 * @brief Return the clamped value between 0.0f and max.
 *
 * It is assumed that  @c max is not a NaN value. If @c a is NaN then zero will
 * be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 clampz(float maxv, vfloat4 a)
{
	// Do not reorder - second operand will return if either is NaN
	return min(max(a, vfloat4::zero()), maxv);
}

/**
 * @brief Return the clamped value between 0.0f and 1.0f.
 *
 * If @c a is NaN then zero will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 clampzo(vfloat4 a)
{
	// Do not reorder - second operand will return if either is NaN
	return min(max(a, vfloat4::zero()), 1.0f);
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE float hmin_s(vfloat4 a)
{
	return hmin(a).lane<0>();
}

/**
 * @brief Return the horizontal min of RGB vector lanes as a scalar.
 */
ASTCENC_SIMD_INLINE float hmin_rgb_s(vfloat4 a)
{
	a.set_lane<3>(a.lane<0>());
	return hmin_s(a);
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE float hmax_s(vfloat4 a)
{
	return hmax(a).lane<0>();
}

/**
 * @brief Accumulate lane-wise sums for a vector.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat4 a)
{
	accum = accum + a;
}

/**
 * @brief Accumulate lane-wise sums for a masked vector.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat4 a, vmask4 m)
{
	a = select(vfloat4::zero(), a, m);
	haccumulate(accum, a);
}

/**
 * @brief Return the horizontal sum of RGB vector lanes as a scalar.
 */
ASTCENC_SIMD_INLINE float hadd_rgb_s(vfloat4 a)
{
	return a.lane<0>() + a.lane<1>() + a.lane<2>();
}

#if !defined(ASTCENC_USE_NATIVE_DOT_PRODUCT)

/**
 * @brief Return the dot product for the full 4 lanes, returning scalar.
 */
ASTCENC_SIMD_INLINE float dot_s(vfloat4 a, vfloat4 b)
{
	vfloat4 m = a * b;
	return hadd_s(m);
}

/**
 * @brief Return the dot product for the full 4 lanes, returning vector.
 */
ASTCENC_SIMD_INLINE vfloat4 dot(vfloat4 a, vfloat4 b)
{
	vfloat4 m = a * b;
	return vfloat4(hadd_s(m));
}

/**
 * @brief Return the dot product for the bottom 3 lanes, returning scalar.
 */
ASTCENC_SIMD_INLINE float dot3_s(vfloat4 a, vfloat4 b)
{
	vfloat4 m = a * b;
	return hadd_rgb_s(m);
}

/**
 * @brief Return the dot product for the bottom 3 lanes, returning vector.
 */
ASTCENC_SIMD_INLINE vfloat4 dot3(vfloat4 a, vfloat4 b)
{
	vfloat4 m = a * b;
	float d3 = hadd_rgb_s(m);
	return vfloat4(d3, d3, d3, 0.0f);
}

#endif

#if !defined(ASTCENC_USE_NATIVE_POPCOUNT)

/**
 * @brief Population bit count.
 *
 * @param v   The value to population count.
 *
 * @return The number of 1 bits.
 */
static inline int popcount(uint64_t v)
{
	uint64_t mask1 = 0x5555555555555555ULL;
	uint64_t mask2 = 0x3333333333333333ULL;
	uint64_t mask3 = 0x0F0F0F0F0F0F0F0FULL;
	v -= (v >> 1) & mask1;
	v = (v & mask2) + ((v >> 2) & mask2);
	v += v >> 4;
	v &= mask3;
	v *= 0x0101010101010101ULL;
	v >>= 56;
	return static_cast<int>(v);
}

#endif

/**
 * @brief Apply signed bit transfer.
 *
 * @param input0   The first encoded endpoint.
 * @param input1   The second encoded endpoint.
 */
static ASTCENC_SIMD_INLINE void bit_transfer_signed(
	vint4& input0,
	vint4& input1
) {
	input1 = lsr<1>(input1) | (input0 & 0x80);
	input0 = lsr<1>(input0) & 0x3F;

	vmask4 mask = (input0 & 0x20) != vint4::zero();
	input0 = select(input0, input0 - 0x40, mask);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void print(vint4 a)
{
	ASTCENC_ALIGNAS int v[4];
	storea(a, v);
	printf("v4_i32:\n  %8d %8d %8d %8d\n",
	       v[0], v[1], v[2], v[3]);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void printx(vint4 a)
{
	ASTCENC_ALIGNAS int v[4];
	storea(a, v);
	printf("v4_i32:\n  %08x %08x %08x %08x\n",
	       v[0], v[1], v[2], v[3]);
}

/**
 * @brief Debug function to print a vector of floats.
 */
ASTCENC_SIMD_INLINE void print(vfloat4 a)
{
	ASTCENC_ALIGNAS float v[4];
	storea(a, v);
	printf("v4_f32:\n  %0.4f %0.4f %0.4f %0.4f\n",
	       static_cast<double>(v[0]), static_cast<double>(v[1]),
	       static_cast<double>(v[2]), static_cast<double>(v[3]));
}

/**
 * @brief Debug function to print a vector of masks.
 */
ASTCENC_SIMD_INLINE void print(vmask4 a)
{
	print(select(vint4(0), vint4(1), a));
}

#endif // #ifndef ASTC_VECMATHLIB_COMMON_4_H_INCLUDED

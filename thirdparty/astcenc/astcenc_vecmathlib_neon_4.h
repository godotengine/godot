// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2019-2024 Arm Limited
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
 * @brief 4x32-bit vectors, implemented using Armv8-A NEON.
 *
 * This module implements 4-wide 32-bit float, int, and mask vectors for
 * Armv8-A NEON.
 *
 * There is a baseline level of functionality provided by all vector widths and
 * implementations. This is implemented using identical function signatures,
 * modulo data type, so we can use them as substitutable implementations in VLA
 * code.
 *
 * The 4-wide vectors are also used as a fixed-width type, and significantly
 * extend the functionality above that available to VLA code.
 */

#ifndef ASTC_VECMATHLIB_NEON_4_H_INCLUDED
#define ASTC_VECMATHLIB_NEON_4_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>
#include <cstring>

// ============================================================================
// vfloat4 data type
// ============================================================================

/**
 * @brief Data type for 4-wide floats.
 */
struct vfloat4
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vfloat4() = default;

	/**
	 * @brief Construct from 4 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat4(const float *p)
	{
		m = vld1q_f32(p);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat4(float a)
	{
		m = vdupq_n_f32(a);
	}

	/**
	 * @brief Construct from 4 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat4(float a, float b, float c, float d)
	{
		float v[4] { a, b, c, d };
		m = vld1q_f32(v);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat4(float32x4_t a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar value of a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE float lane() const
	{
		return vgetq_lane_f32(m, l);
	}

	/**
	 * @brief Set the scalar value of a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE void set_lane(float a)
	{
		m = vsetq_lane_f32(a, m, l);
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vfloat4 zero()
	{
		return vfloat4(0.0f);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat4 load1(const float* p)
	{
		return vfloat4(vld1q_dup_f32(p));
	}

	/**
	 * @brief Factory that returns a vector loaded from 16B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat4 loada(const float* p)
	{
		return vfloat4(vld1q_f32(p));
	}

	/**
	 * @brief Return a swizzled float 2.
	 */
	template <int l0, int l1> ASTCENC_SIMD_INLINE vfloat4 swz() const
	{
		return vfloat4(lane<l0>(), lane<l1>(), 0.0f, 0.0f);
	}

	/**
	 * @brief Return a swizzled float 3.
	 */
	template <int l0, int l1, int l2> ASTCENC_SIMD_INLINE vfloat4 swz() const
	{
		return vfloat4(lane<l0>(), lane<l1>(), lane<l2>(), 0.0f);
	}

	/**
	 * @brief Return a swizzled float 4.
	 */
	template <int l0, int l1, int l2, int l3> ASTCENC_SIMD_INLINE vfloat4 swz() const
	{
		return vfloat4(lane<l0>(), lane<l1>(), lane<l2>(), lane<l3>());
	}

	/**
	 * @brief The vector ...
	 */
	float32x4_t m;
};

// ============================================================================
// vint4 data type
// ============================================================================

/**
 * @brief Data type for 4-wide ints.
 */
struct vint4
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vint4() = default;

	/**
	 * @brief Construct from 4 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vint4(const int *p)
	{
		m = vld1q_s32(p);
	}

	/**
	 * @brief Construct from 4 uint8_t loaded from an unaligned address.
	 */
	ASTCENC_SIMD_INLINE explicit vint4(const uint8_t *p)
	{
#if ASTCENC_SVE == 0
	// Cast is safe - NEON loads are allowed to be unaligned
	uint32x2_t t8 = vld1_dup_u32(reinterpret_cast<const uint32_t*>(p));
	uint16x4_t t16 = vget_low_u16(vmovl_u8(vreinterpret_u8_u32(t8)));
	m = vreinterpretq_s32_u32(vmovl_u16(t16));
#else
	svint32_t data = svld1ub_s32(svptrue_pat_b32(SV_VL4), p);
	m = svget_neonq(data);
#endif
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vint4(int a)
	{
		m = vdupq_n_s32(a);
	}

	/**
	 * @brief Construct from 4 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint4(int a, int b, int c, int d)
	{
		int v[4] { a, b, c, d };
		m = vld1q_s32(v);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint4(int32x4_t a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar from a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE int lane() const
	{
		return vgetq_lane_s32(m, l);
	}

	/**
	 * @brief Set the scalar value of a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE void set_lane(int a)
	{
		m = vsetq_lane_s32(a, m, l);
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vint4 zero()
	{
		return vint4(0);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vint4 load1(const int* p)
	{
		return vint4(*p);
	}

	/**
	 * @brief Factory that returns a vector loaded from unaligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint4 load(const uint8_t* p)
	{
		vint4 data;
		std::memcpy(&data.m, p, 4 * sizeof(int));
		return data;
	}

	/**
	 * @brief Factory that returns a vector loaded from 16B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint4 loada(const int* p)
	{
		return vint4(p);
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vint4 lane_id()
	{
		alignas(16) static const int data[4] { 0, 1, 2, 3 };
		return vint4(vld1q_s32(data));
	}

	/**
	 * @brief The vector ...
	 */
	int32x4_t m;
};

// ============================================================================
// vmask4 data type
// ============================================================================

/**
 * @brief Data type for 4-wide control plane masks.
 */
struct vmask4
{
	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask4(uint32x4_t a)
	{
		m = a;
	}

#if !defined(_MSC_VER)
	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask4(int32x4_t a)
	{
		m = vreinterpretq_u32_s32(a);
	}
#endif

	/**
	 * @brief Construct from 1 scalar value.
	 */
	ASTCENC_SIMD_INLINE explicit vmask4(bool a)
	{
		m = vreinterpretq_u32_s32(vdupq_n_s32(a == true ? -1 : 0));
	}

	/**
	 * @brief Construct from 4 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask4(bool a, bool b, bool c, bool d)
	{
		int v[4] {
			a == true ? -1 : 0,
			b == true ? -1 : 0,
			c == true ? -1 : 0,
			d == true ? -1 : 0
		};

		int32x4_t ms = vld1q_s32(v);
		m = vreinterpretq_u32_s32(ms);
	}

	/**
	 * @brief Get the scalar from a single lane.
	 */
	template <int32_t l> ASTCENC_SIMD_INLINE bool lane() const
	{
		return vgetq_lane_u32(m, l) != 0;
	}

	/**
	 * @brief The vector ...
	 */
	uint32x4_t m;
};

// ============================================================================
// vmask4 operators and functions
// ============================================================================

/**
 * @brief Overload: mask union (or).
 */
ASTCENC_SIMD_INLINE vmask4 operator|(vmask4 a, vmask4 b)
{
	return vmask4(vorrq_u32(a.m, b.m));
}

/**
 * @brief Overload: mask intersect (and).
 */
ASTCENC_SIMD_INLINE vmask4 operator&(vmask4 a, vmask4 b)
{
	return vmask4(vandq_u32(a.m, b.m));
}

/**
 * @brief Overload: mask difference (xor).
 */
ASTCENC_SIMD_INLINE vmask4 operator^(vmask4 a, vmask4 b)
{
	return vmask4(veorq_u32(a.m, b.m));
}

/**
 * @brief Overload: mask invert (not).
 */
ASTCENC_SIMD_INLINE vmask4 operator~(vmask4 a)
{
	return vmask4(vmvnq_u32(a.m));
}

/**
 * @brief Return a 4-bit mask code indicating mask status.
 *
 * bit0 = lane 0
 */
ASTCENC_SIMD_INLINE unsigned int mask(vmask4 a)
{
	static const int shifta[4] { 0, 1, 2, 3 };
	static const int32x4_t shift = vld1q_s32(shifta);

	uint32x4_t tmp = vshrq_n_u32(a.m, 31);
	return vaddvq_u32(vshlq_u32(tmp, shift));
}

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask4 a)
{
	return vmaxvq_u32(a.m) != 0;
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask4 a)
{
	return vminvq_u32(a.m) != 0;
}

// ============================================================================
// vint4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vint4 operator+(vint4 a, vint4 b)
{
	return vint4(vaddq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vint4 operator-(vint4 a, vint4 b)
{
	return vint4(vsubq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vint4 operator*(vint4 a, vint4 b)
{
	return vint4(vmulq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector bit invert.
 */
ASTCENC_SIMD_INLINE vint4 operator~(vint4 a)
{
	return vint4(vmvnq_s32(a.m));
}

/**
 * @brief Overload: vector by vector bitwise or.
 */
ASTCENC_SIMD_INLINE vint4 operator|(vint4 a, vint4 b)
{
	return vint4(vorrq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise and.
 */
ASTCENC_SIMD_INLINE vint4 operator&(vint4 a, vint4 b)
{
	return vint4(vandq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise xor.
 */
ASTCENC_SIMD_INLINE vint4 operator^(vint4 a, vint4 b)
{
	return vint4(veorq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask4 operator==(vint4 a, vint4 b)
{
	return vmask4(vceqq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask4 operator!=(vint4 a, vint4 b)
{
	return ~vmask4(vceqq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask4 operator<(vint4 a, vint4 b)
{
	return vmask4(vcltq_s32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask4 operator>(vint4 a, vint4 b)
{
	return vmask4(vcgtq_s32(a.m, b.m));
}

/**
 * @brief Logical shift left.
 */
template <int s> ASTCENC_SIMD_INLINE vint4 lsl(vint4 a)
{
	return vint4(vshlq_s32(a.m, vdupq_n_s32(s)));
}

/**
 * @brief Logical shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint4 lsr(vint4 a)
{
	uint32x4_t ua = vreinterpretq_u32_s32(a.m);
	ua = vshlq_u32(ua, vdupq_n_s32(-s));
	return vint4(vreinterpretq_s32_u32(ua));
}

/**
 * @brief Arithmetic shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint4 asr(vint4 a)
{
	return vint4(vshlq_s32(a.m, vdupq_n_s32(-s)));
}

/**
 * @brief Return the min vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint4 min(vint4 a, vint4 b)
{
	return vint4(vminq_s32(a.m, b.m));
}

/**
 * @brief Return the max vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint4 max(vint4 a, vint4 b)
{
	return vint4(vmaxq_s32(a.m, b.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vint4 hmin(vint4 a)
{
	return vint4(vminvq_s32(a.m));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vint4 hmax(vint4 a)
{
	return vint4(vmaxvq_s32(a.m));
}

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vint4 a, int* p)
{
	vst1q_s32(p, a.m);
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint4 a, int* p)
{
	vst1q_s32(p, a.m);
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint4 a, uint8_t* p)
{
	std::memcpy(p, &a.m, sizeof(int) * 4);
}

/**
 * @brief Store lowest N (vector width) bytes into an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_nbytes(vint4 a, uint8_t* p)
{
	vst1q_lane_s32(reinterpret_cast<int32_t*>(p), a.m, 0);
}

/**
 * @brief Pack and store low 8 bits of each vector lane.
 */
ASTCENC_SIMD_INLINE void pack_and_store_low_bytes(vint4 a, uint8_t* data)
{
	alignas(16) uint8_t shuf[16] {
		0, 4, 8, 12,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0
	};
	uint8x16_t idx = vld1q_u8(shuf);
	int8x16_t av = vreinterpretq_s8_s32(a.m);
	a = vint4(vreinterpretq_s32_s8(vqtbl1q_s8(av, idx)));
	store_nbytes(a, data);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vint4 select(vint4 a, vint4 b, vmask4 cond)
{
	return vint4(vbslq_s32(cond.m, b.m, a.m));
}

// ============================================================================
// vfloat4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vfloat4 operator+(vfloat4 a, vfloat4 b)
{
	return vfloat4(vaddq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vfloat4 operator-(vfloat4 a, vfloat4 b)
{
	return vfloat4(vsubq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat4 operator*(vfloat4 a, vfloat4 b)
{
	return vfloat4(vmulq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector division.
 */
ASTCENC_SIMD_INLINE vfloat4 operator/(vfloat4 a, vfloat4 b)
{
	return vfloat4(vdivq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask4 operator==(vfloat4 a, vfloat4 b)
{
	return vmask4(vceqq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask4 operator!=(vfloat4 a, vfloat4 b)
{
	return vmask4(vmvnq_u32(vceqq_f32(a.m, b.m)));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask4 operator<(vfloat4 a, vfloat4 b)
{
	return vmask4(vcltq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask4 operator>(vfloat4 a, vfloat4 b)
{
	return vmask4(vcgtq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than or equal.
 */
ASTCENC_SIMD_INLINE vmask4 operator<=(vfloat4 a, vfloat4 b)
{
	return vmask4(vcleq_f32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than or equal.
 */
ASTCENC_SIMD_INLINE vmask4 operator>=(vfloat4 a, vfloat4 b)
{
	return vmask4(vcgeq_f32(a.m, b.m));
}

/**
 * @brief Return the min vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 min(vfloat4 a, vfloat4 b)
{
	// Do not reorder - second operand will return if either is NaN
	return vfloat4(vminnmq_f32(a.m, b.m));
}

/**
 * @brief Return the max vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat4 max(vfloat4 a, vfloat4 b)
{
	// Do not reorder - second operand will return if either is NaN
	return vfloat4(vmaxnmq_f32(a.m, b.m));
}

/**
 * @brief Return the absolute value of the float vector.
 */
ASTCENC_SIMD_INLINE vfloat4 abs(vfloat4 a)
{
	float32x4_t zero = vdupq_n_f32(0.0f);
	float32x4_t inv = vsubq_f32(zero, a.m);
	return vfloat4(vmaxq_f32(a.m, inv));
}

/**
 * @brief Return a float rounded to the nearest integer value.
 */
ASTCENC_SIMD_INLINE vfloat4 round(vfloat4 a)
{
	return vfloat4(vrndnq_f32(a.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat4 hmin(vfloat4 a)
{
	return vfloat4(vminvq_f32(a.m));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat4 hmax(vfloat4 a)
{
	return vfloat4(vmaxvq_f32(a.m));
}

/**
 * @brief Return the horizontal sum of a vector.
 */
ASTCENC_SIMD_INLINE float hadd_s(vfloat4 a)
{
	// Perform halving add to ensure invariance; we cannot use vaddqv as this
	// does (0 + 1 + 2 + 3) which is not invariant with x86 (0 + 2) + (1 + 3).
	float32x2_t t = vadd_f32(vget_high_f32(a.m), vget_low_f32(a.m));
	return vget_lane_f32(vpadd_f32(t, t), 0);
}

/**
 * @brief Return the sqrt of the lanes in the vector.
 */
ASTCENC_SIMD_INLINE vfloat4 sqrt(vfloat4 a)
{
	return vfloat4(vsqrtq_f32(a.m));
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat4 select(vfloat4 a, vfloat4 b, vmask4 cond)
{
	return vfloat4(vbslq_f32(cond.m, b.m, a.m));
}

/**
 * @brief Load a vector of gathered results from an array;
 */
ASTCENC_SIMD_INLINE vfloat4 gatherf(const float* base, vint4 indices)
{
#if ASTCENC_SVE == 0
	alignas(16) int idx[4];
	storea(indices, idx);
	alignas(16) float vals[4];
	vals[0] = base[idx[0]];
	vals[1] = base[idx[1]];
	vals[2] = base[idx[2]];
	vals[3] = base[idx[3]];
	return vfloat4(vals);
#else
	svint32_t offsets = svset_neonq_s32(svundef_s32(), indices.m);
	svfloat32_t data = svld1_gather_s32index_f32(svptrue_pat_b32(SV_VL4), base, offsets);
	return vfloat4(svget_neonq_f32(data));
#endif
}

/**
 * @brief Load a vector of gathered results from an array using byte indices from memory
 */
template<>
ASTCENC_SIMD_INLINE vfloat4 gatherf_byte_inds<vfloat4>(const float* base, const uint8_t* indices)
{
#if ASTCENC_SVE == 0
	alignas(16) float vals[4];
	vals[0] = base[indices[0]];
	vals[1] = base[indices[1]];
	vals[2] = base[indices[2]];
	vals[3] = base[indices[3]];
	return vfloat4(vals);
#else
	svint32_t offsets = svld1ub_s32(svptrue_pat_b32(SV_VL4), indices);
	svfloat32_t data = svld1_gather_s32index_f32(svptrue_pat_b32(SV_VL4), base, offsets);
	return vfloat4(svget_neonq_f32(data));
#endif
}
/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vfloat4 a, float* p)
{
	vst1q_f32(p, a.m);
}

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vfloat4 a, float* p)
{
	vst1q_f32(p, a.m);
}

/**
 * @brief Return a integer value for a float vector, using truncation.
 */
ASTCENC_SIMD_INLINE vint4 float_to_int(vfloat4 a)
{
	return vint4(vcvtq_s32_f32(a.m));
}

/**
 * @brief Return a integer value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint4 float_to_int_rtn(vfloat4 a)
{
	a = a + vfloat4(0.5f);
	return vint4(vcvtq_s32_f32(a.m));
}

/**
 * @brief Return a float value for an integer vector.
 */
ASTCENC_SIMD_INLINE vfloat4 int_to_float(vint4 a)
{
	return vfloat4(vcvtq_f32_s32(a.m));
}

/**
 * @brief Return a float16 value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint4 float_to_float16(vfloat4 a)
{
	// Generate float16 value
	float16x4_t f16 = vcvt_f16_f32(a.m);

	// Convert each 16-bit float pattern to a 32-bit pattern
	uint16x4_t u16 = vreinterpret_u16_f16(f16);
	uint32x4_t u32 = vmovl_u16(u16);
	return vint4(vreinterpretq_s32_u32(u32));
}

/**
 * @brief Return a float16 value for a float scalar, using round-to-nearest.
 */
static inline uint16_t float_to_float16(float a)
{
	vfloat4 av(a);
	return static_cast<uint16_t>(float_to_float16(av).lane<0>());
}

/**
 * @brief Return a float value for a float16 vector.
 */
ASTCENC_SIMD_INLINE vfloat4 float16_to_float(vint4 a)
{
	// Convert each 32-bit float pattern to a 16-bit pattern
	uint32x4_t u32 = vreinterpretq_u32_s32(a.m);
	uint16x4_t u16 = vmovn_u32(u32);
	float16x4_t f16 = vreinterpret_f16_u16(u16);

	// Generate float16 value
	return vfloat4(vcvt_f32_f16(f16));
}

/**
 * @brief Return a float value for a float16 scalar.
 */
ASTCENC_SIMD_INLINE float float16_to_float(uint16_t a)
{
	vint4 av(a);
	return float16_to_float(av).lane<0>();
}

/**
 * @brief Return a float value as an integer bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the first half of that flip.
 */
ASTCENC_SIMD_INLINE vint4 float_as_int(vfloat4 a)
{
	return vint4(vreinterpretq_s32_f32(a.m));
}

/**
 * @brief Return a integer value as a float bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the second half of that flip.
 */
ASTCENC_SIMD_INLINE vfloat4 int_as_float(vint4 v)
{
	return vfloat4(vreinterpretq_f32_s32(v.m));
}

/*
 * Table structure for a 16x 8-bit entry table.
 */
struct vtable4_16x8 {
	uint8x16_t t0;
};

/*
 * Table structure for a 32x 8-bit entry table.
 */
struct vtable4_32x8 {
	uint8x16x2_t t01;
};

/*
 * Table structure for a 64x 8-bit entry table.
 */
struct vtable4_64x8 {
	uint8x16x4_t t0123;
};

/**
 * @brief Prepare a vtable lookup table for 16x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable4_16x8& table,
	const uint8_t* data
) {
	table.t0 = vld1q_u8(data);
}

/**
 * @brief Prepare a vtable lookup table for 32x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable4_32x8& table,
	const uint8_t* data
) {
	table.t01 = uint8x16x2_t {
		vld1q_u8(data),
		vld1q_u8(data + 16)
	};
}

/**
 * @brief Prepare a vtable lookup table 64x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable4_64x8& table,
	const uint8_t* data
) {
	table.t0123 = uint8x16x4_t {
		vld1q_u8(data),
		vld1q_u8(data + 16),
		vld1q_u8(data + 32),
		vld1q_u8(data + 48)
	};
}

/**
 * @brief Perform a vtable lookup in a 16x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint4 vtable_lookup_32bit(
	const vtable4_16x8& tbl,
	vint4 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	int32x4_t idx_masked = vorrq_s32(idx.m, vdupq_n_s32(0xFFFFFF00));
	uint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);

	return vint4(vreinterpretq_s32_u8(vqtbl1q_u8(tbl.t0, idx_bytes)));
}

/**
 * @brief Perform a vtable lookup in a 32x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint4 vtable_lookup_32bit(
	const vtable4_32x8& tbl,
	vint4 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	int32x4_t idx_masked = vorrq_s32(idx.m, vdupq_n_s32(0xFFFFFF00));
	uint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);

	return vint4(vreinterpretq_s32_u8(vqtbl2q_u8(tbl.t01, idx_bytes)));
}

/**
 * @brief Perform a vtable lookup in a 64x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint4 vtable_lookup_32bit(
	const vtable4_64x8& tbl,
	vint4 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	int32x4_t idx_masked = vorrq_s32(idx.m, vdupq_n_s32(0xFFFFFF00));
	uint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);

	return vint4(vreinterpretq_s32_u8(vqtbl4q_u8(tbl.t0123, idx_bytes)));
}

/**
 * @brief Return a vector of interleaved RGBA data.
 *
 * Input vectors have the value stored in the bottom 8 bits of each lane,
 * with high  bits set to zero.
 *
 * Output vector stores a single RGBA texel packed in each lane.
 */
ASTCENC_SIMD_INLINE vint4 interleave_rgba8(vint4 r, vint4 g, vint4 b, vint4 a)
{
	return r + lsl<8>(g) + lsl<16>(b) + lsl<24>(a);
}

/**
 * @brief Store a single vector lane to an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_lane(uint8_t* base, int data)
{
	std::memcpy(base, &data, sizeof(int));
}

/**
 * @brief Store a vector, skipping masked lanes.
 *
 * All masked lanes must be at the end of vector, after all non-masked lanes.
 */
ASTCENC_SIMD_INLINE void store_lanes_masked(uint8_t* base, vint4 data, vmask4 mask)
{
	if (mask.lane<3>())
	{
		store(data, base);
	}
	else if (mask.lane<2>() != 0.0f)
	{
		store_lane(base + 0, data.lane<0>());
		store_lane(base + 4, data.lane<1>());
		store_lane(base + 8, data.lane<2>());
	}
	else if (mask.lane<1>() != 0.0f)
	{
		store_lane(base + 0, data.lane<0>());
		store_lane(base + 4, data.lane<1>());
	}
	else if (mask.lane<0>() != 0.0f)
	{
		store_lane(base + 0, data.lane<0>());
	}
}

#define ASTCENC_USE_NATIVE_POPCOUNT 1

/**
 * @brief Population bit count.
 *
 * @param v   The value to population count.
 *
 * @return The number of 1 bits.
 */
ASTCENC_SIMD_INLINE int popcount(uint64_t v)
{
	return static_cast<int>(vaddlv_u8(vcnt_u8(vcreate_u8(v))));
}

#endif // #ifndef ASTC_VECMATHLIB_NEON_4_H_INCLUDED

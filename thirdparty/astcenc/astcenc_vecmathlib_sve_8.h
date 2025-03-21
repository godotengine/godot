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
 * @brief 8x32-bit vectors, implemented using SVE.
 *
 * This module implements 8-wide 32-bit float, int, and mask vectors for Arm
 * SVE.
 *
 * There is a baseline level of functionality provided by all vector widths and
 * implementations. This is implemented using identical function signatures,
 * modulo data type, so we can use them as substitutable implementations in VLA
 * code.
 */

#ifndef ASTC_VECMATHLIB_SVE_8_H_INCLUDED
#define ASTC_VECMATHLIB_SVE_8_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

typedef svbool_t svbool_8_t __attribute__((arm_sve_vector_bits(256)));
typedef svuint8_t svuint8_8_t __attribute__((arm_sve_vector_bits(256)));
typedef svuint16_t svuint16_8_t __attribute__((arm_sve_vector_bits(256)));
typedef svuint32_t svuint32_8_t __attribute__((arm_sve_vector_bits(256)));
typedef svint32_t svint32_8_t __attribute__((arm_sve_vector_bits(256)));
typedef svfloat32_t svfloat32_8_t __attribute__((arm_sve_vector_bits(256)));

// ============================================================================
// vfloat8 data type
// ============================================================================

/**
 * @brief Data type for 8-wide floats.
 */
struct vfloat8
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vfloat8() = default;

	/**
	 * @brief Construct from 8 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(const float *p)
	{
		m = svld1_f32(svptrue_b32(), p);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(float a)
	{
		m = svdup_f32(a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(svfloat32_8_t a)
	{
		m = a;
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 zero()
	{
		return vfloat8(0.0f);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 load1(const float* p)
	{
		return vfloat8(*p);
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 loada(const float* p)
	{
		return vfloat8(p);
	}

	/**
	 * @brief The vector ...
	 */
	svfloat32_8_t m;
};

// ============================================================================
// vint8 data type
// ============================================================================

/**
 * @brief Data type for 8-wide ints.
 */
struct vint8
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vint8() = default;

	/**
	 * @brief Construct from 8 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(const int *p)
	{
		m = svld1_s32(svptrue_b32(), p);
	}

	/**
	 * @brief Construct from 8 uint8_t loaded from an unaligned address.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(const uint8_t *p)
	{
		// Load 8-bit values and expand to 32-bits
		m = svld1ub_s32(svptrue_b32(), p);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(int a)
	{
		m = svdup_s32(a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(svint32_8_t a)
	{
		m = a;
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vint8 zero()
	{
		return vint8(0.0f);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vint8 load1(const int* p)
	{
		return vint8(*p);
	}

	/**
	 * @brief Factory that returns a vector loaded from unaligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint8 load(const uint8_t* p)
	{
		svuint8_8_t data = svld1_u8(svptrue_b8(), p);
		return vint8(svreinterpret_s32_u8(data));
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint8 loada(const int* p)
	{
		return vint8(p);
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vint8 lane_id()
	{
		return vint8(svindex_s32(0, 1));
	}

	/**
	 * @brief The vector ...
	 */
	 svint32_8_t m;
};

// ============================================================================
// vmask8 data type
// ============================================================================

/**
 * @brief Data type for 8-wide control plane masks.
 */
struct vmask8
{
	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask8(svbool_8_t a)
	{
		m = a;
	}

	/**
	 * @brief Construct from 1 scalar value.
	 */
	ASTCENC_SIMD_INLINE explicit vmask8(bool a)
	{
		m = svdup_b32(a);
	}

	/**
	 * @brief The vector ...
	 */
	svbool_8_t m;
};

// ============================================================================
// vmask8 operators and functions
// ============================================================================

/**
 * @brief Overload: mask union (or).
 */
ASTCENC_SIMD_INLINE vmask8 operator|(vmask8 a, vmask8 b)
{
	return vmask8(svorr_z(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: mask intersect (and).
 */
ASTCENC_SIMD_INLINE vmask8 operator&(vmask8 a, vmask8 b)
{
	return vmask8(svand_z(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: mask difference (xor).
 */
ASTCENC_SIMD_INLINE vmask8 operator^(vmask8 a, vmask8 b)
{
	return vmask8(sveor_z(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: mask invert (not).
 */
ASTCENC_SIMD_INLINE vmask8 operator~(vmask8 a)
{
	return vmask8(svnot_z(svptrue_b32(), a.m));
}

/**
 * @brief Return a 8-bit mask code indicating mask status.
 *
 * bit0 = lane 0
 */
ASTCENC_SIMD_INLINE unsigned int mask(vmask8 a)
{
	alignas(32) const int shifta[8] { 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80 };
	svint32_8_t template_vals = svld1_s32(svptrue_b32(), shifta);
	svint32_8_t active_vals = svsel_s32(a.m, template_vals, svdup_s32(0));
	return static_cast<unsigned int>(svaddv_s32(svptrue_b32(), active_vals));
}

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask8 a)
{
	return svptest_any(svptrue_b32(), a.m);
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask8 a)
{
	return !svptest_any(svptrue_b32(), (~a).m);
}

// ============================================================================
// vint8 operators and functions
// ============================================================================
/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vint8 operator+(vint8 a, vint8 b)
{
	return vint8(svadd_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vint8& operator+=(vint8& a, const vint8& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vint8 operator-(vint8 a, vint8 b)
{
	return vint8(svsub_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vint8 operator*(vint8 a, vint8 b)
{
	return vint8(svmul_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector bit invert.
 */
ASTCENC_SIMD_INLINE vint8 operator~(vint8 a)
{
	return vint8(svnot_s32_x(svptrue_b32(), a.m));
}

/**
 * @brief Overload: vector by vector bitwise or.
 */
ASTCENC_SIMD_INLINE vint8 operator|(vint8 a, vint8 b)
{
	return vint8(svorr_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise and.
 */
ASTCENC_SIMD_INLINE vint8 operator&(vint8 a, vint8 b)
{
	return vint8(svand_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise xor.
 */
ASTCENC_SIMD_INLINE vint8 operator^(vint8 a, vint8 b)
{
	return vint8(sveor_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask8 operator==(vint8 a, vint8 b)
{
	return vmask8(svcmpeq_s32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask8 operator!=(vint8 a, vint8 b)
{
	return vmask8(svcmpne_s32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask8 operator<(vint8 a, vint8 b)
{
	return vmask8(svcmplt_s32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask8 operator>(vint8 a, vint8 b)
{
	return vmask8(svcmpgt_s32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Logical shift left.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 lsl(vint8 a)
{
	return vint8(svlsl_n_s32_x(svptrue_b32(), a.m, s));
}

/**
 * @brief Arithmetic shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 asr(vint8 a)
{
	return vint8(svasr_n_s32_x(svptrue_b32(), a.m, s));
}

/**
 * @brief Logical shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 lsr(vint8 a)
{
	svuint32_8_t r = svreinterpret_u32_s32(a.m);
	r = svlsr_n_u32_x(svptrue_b32(), r, s);
	return vint8(svreinterpret_s32_u32(r));
}

/**
 * @brief Return the min vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint8 min(vint8 a, vint8 b)
{
	return vint8(svmin_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Return the max vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint8 max(vint8 a, vint8 b)
{
	return vint8(svmax_s32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vint8 hmin(vint8 a)
{
	return vint8(svminv_s32(svptrue_b32(), a.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE int hmin_s(vint8 a)
{
	return svminv_s32(svptrue_b32(), a.m);
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vint8 hmax(vint8 a)
{
	return vint8(svmaxv_s32(svptrue_b32(), a.m));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE int hmax_s(vint8 a)
{
	return svmaxv_s32(svptrue_b32(), a.m);
}

/**
 * @brief Generate a vint8 from a size_t.
 */
 ASTCENC_SIMD_INLINE vint8 vint8_from_size(size_t a)
 {
	assert(a <= std::numeric_limits<int>::max());
	return vint8(static_cast<int>(a));
 }

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vint8 a, int* p)
{
	svst1_s32(svptrue_b32(), p, a.m);
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint8 a, int* p)
{
	svst1_s32(svptrue_b32(), p, a.m);
}

/**
 * @brief Store lowest N (vector width) bytes into an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_nbytes(vint8 a, uint8_t* p)
{
	svuint8_8_t r = svreinterpret_u8_s32(a.m);
	svst1_u8(svptrue_pat_b8(SV_VL8), p, r);
}

/**
 * @brief Pack low 8 bits of N (vector width) lanes into bottom of vector.
 */
ASTCENC_SIMD_INLINE void pack_and_store_low_bytes(vint8 v, uint8_t* p)
{
	svuint32_8_t data = svreinterpret_u32_s32(v.m);
	svst1b_u32(svptrue_b32(), p, data);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vint8 select(vint8 a, vint8 b, vmask8 cond)
{
	return vint8(svsel_s32(cond.m, b.m, a.m));
}

// ============================================================================
// vfloat8 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vfloat8 operator+(vfloat8 a, vfloat8 b)
{
	return vfloat8(svadd_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vfloat8& operator+=(vfloat8& a, const vfloat8& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vfloat8 operator-(vfloat8 a, vfloat8 b)
{
	return vfloat8(svsub_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(vfloat8 a, vfloat8 b)
{
	return vfloat8(svmul_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(vfloat8 a, float b)
{
	return vfloat8(svmul_f32_x(svptrue_b32(), a.m, svdup_f32(b)));
}

/**
 * @brief Overload: scalar by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(float a, vfloat8 b)
{
	return vfloat8(svmul_f32_x(svptrue_b32(), svdup_f32(a), b.m));
}

/**
 * @brief Overload: vector by vector division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(vfloat8 a, vfloat8 b)
{
	return vfloat8(svdiv_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by scalar division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(vfloat8 a, float b)
{
	return vfloat8(svdiv_f32_x(svptrue_b32(), a.m, svdup_f32(b)));
}

/**
 * @brief Overload: scalar by vector division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(float a, vfloat8 b)
{
	return vfloat8(svdiv_f32_x(svptrue_b32(), svdup_f32(a), b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask8 operator==(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmpeq_f32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask8 operator!=(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmpne_f32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask8 operator<(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmplt_f32(svptrue_b32(), a.m, b.m));;
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask8 operator>(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmpgt_f32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than or equal.
 */
ASTCENC_SIMD_INLINE vmask8 operator<=(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmple_f32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Overload: vector by vector greater than or equal.
 */
ASTCENC_SIMD_INLINE vmask8 operator>=(vfloat8 a, vfloat8 b)
{
	return vmask8(svcmpge_f32(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Return the min vector of two vectors.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat8 min(vfloat8 a, vfloat8 b)
{
	return vfloat8(svminnm_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Return the min vector of a vector and a scalar.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat8 min(vfloat8 a, float b)
{
	return min(a, vfloat8(b));
}

/**
 * @brief Return the max vector of two vectors.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat8 max(vfloat8 a, vfloat8 b)
{
	return vfloat8(svmaxnm_f32_x(svptrue_b32(), a.m, b.m));
}

/**
 * @brief Return the max vector of a vector and a scalar.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat8 max(vfloat8 a, float b)
{
	return max(a, vfloat8(b));
}

/**
 * @brief Return the clamped value between min and max.
 *
 * It is assumed that neither @c min nor @c max are NaN values. If @c a is NaN
 * then @c min will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 clamp(float minv, float maxv, vfloat8 a)
{
	return min(max(a, minv), maxv);
}

/**
 * @brief Return a clamped value between 0.0f and 1.0f.
 *
 * If @c a is NaN then zero will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 clampzo(vfloat8 a)
{
	return clamp(0.0f, 1.0f, a);
}

/**
 * @brief Return the absolute value of the float vector.
 */
ASTCENC_SIMD_INLINE vfloat8 abs(vfloat8 a)
{
	return vfloat8(svabs_f32_x(svptrue_b32(), a.m));
}

/**
 * @brief Return a float rounded to the nearest integer value.
 */
ASTCENC_SIMD_INLINE vfloat8 round(vfloat8 a)
{
	return vfloat8(svrintn_f32_x(svptrue_b32(), a.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat8 hmin(vfloat8 a)
{
	return vfloat8(svminnmv_f32(svptrue_b32(), a.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE float hmin_s(vfloat8 a)
{
	return svminnmv_f32(svptrue_b32(), a.m);
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat8 hmax(vfloat8 a)
{
	return vfloat8(svmaxnmv_f32(svptrue_b32(), a.m));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE float hmax_s(vfloat8 a)
{
	return svmaxnmv_f32(svptrue_b32(), a.m);
}

/**
 * @brief Return the horizontal sum of a vector.
 */
ASTCENC_SIMD_INLINE float hadd_s(vfloat8 a)
{
	// Can't use svaddv - it's not invariant
	vfloat4 lo(svget_neonq_f32(a.m));
	vfloat4 hi(svget_neonq_f32(svext_f32(a.m, a.m, 4)));
	return hadd_s(lo) + hadd_s(hi);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat8 select(vfloat8 a, vfloat8 b, vmask8 cond)
{
	return vfloat8(svsel_f32(cond.m, b.m, a.m));
}

/**
 * @brief Accumulate lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat8 a)
{
	vfloat4 lo(svget_neonq_f32(a.m));
	haccumulate(accum, lo);

	vfloat4 hi(svget_neonq_f32(svext_f32(a.m, a.m, 4)));
	haccumulate(accum, hi);
}

/**
 * @brief Accumulate lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat8& accum, vfloat8 a)
{
	accum += a;
}

/**
 * @brief Accumulate masked lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat8 a, vmask8 m)
{
	a = select(vfloat8::zero(), a, m);
	haccumulate(accum, a);
}

/**
 * @brief Accumulate masked lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat8& accum, vfloat8 a, vmask8 m)
{
	accum.m = svadd_f32_m(m.m, accum.m, a.m);
}

/**
 * @brief Return the sqrt of the lanes in the vector.
 */
ASTCENC_SIMD_INLINE vfloat8 sqrt(vfloat8 a)
{
	return vfloat8(svsqrt_f32_x(svptrue_b32(), a.m));
}

/**
 * @brief Load a vector of gathered results from an array;
 */
ASTCENC_SIMD_INLINE vfloat8 gatherf(const float* base, vint8 indices)
{
	return vfloat8(svld1_gather_s32index_f32(svptrue_b32(), base, indices.m));
}

/**
 * @brief Load a vector of gathered results from an array using byte indices from memory
 */
template<>
ASTCENC_SIMD_INLINE vfloat8 gatherf_byte_inds<vfloat8>(const float* base, const uint8_t* indices)
{
	svint32_t offsets = svld1ub_s32(svptrue_b32(), indices);
	return vfloat8(svld1_gather_s32index_f32(svptrue_b32(), base, offsets));
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vfloat8 a, float* p)
{
	svst1_f32(svptrue_b32(), p, a.m);
}

/**
 * @brief Store a vector to a 32B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vfloat8 a, float* p)
{
	svst1_f32(svptrue_b32(), p, a.m);
}

/**
 * @brief Return a integer value for a float vector, using truncation.
 */
ASTCENC_SIMD_INLINE vint8 float_to_int(vfloat8 a)
{
	return vint8(svcvt_s32_f32_x(svptrue_b32(), a.m));
}

/**
 * @brief Return a integer value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint8 float_to_int_rtn(vfloat8 a)
{
	a = a + vfloat8(0.5f);
	return vint8(svcvt_s32_f32_x(svptrue_b32(), a.m));
}

/**
 * @brief Return a float value for an integer vector.
 */
ASTCENC_SIMD_INLINE vfloat8 int_to_float(vint8 a)
{
	return vfloat8(svcvt_f32_s32_x(svptrue_b32(), a.m));
}

/**
 * @brief Return a float value as an integer bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the first half of that flip.
 */
ASTCENC_SIMD_INLINE vint8 float_as_int(vfloat8 a)
{
	return vint8(svreinterpret_s32_f32(a.m));
}

/**
 * @brief Return a integer value as a float bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the second half of that flip.
 */
ASTCENC_SIMD_INLINE vfloat8 int_as_float(vint8 a)
{
	return vfloat8(svreinterpret_f32_s32(a.m));
}

/*
 * Table structure for a 16x 8-bit entry table.
 */
struct vtable8_16x8 {
	svuint8_8_t t0;
};

/*
 * Table structure for a 32x 8-bit entry table.
 */
struct vtable8_32x8 {
	svuint8_8_t t0;
};

/*
 * Table structure for a 64x 8-bit entry table.
 */
struct vtable8_64x8 {
	svuint8_8_t t0;
	svuint8_8_t t1;
};

/**
 * @brief Prepare a vtable lookup table for 16x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable8_16x8& table,
	const uint8_t* data
) {
	// Top half of register will be zeros
	table.t0 = svld1_u8(svptrue_pat_b8(SV_VL16), data);
}

/**
 * @brief Prepare a vtable lookup table for 32x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable8_32x8& table,
	const uint8_t* data
) {
	table.t0 = svld1_u8(svptrue_b8(), data);
}

/**
 * @brief Prepare a vtable lookup table 64x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable8_64x8& table,
	const uint8_t* data
) {
	table.t0 = svld1_u8(svptrue_b8(), data);
	table.t1 = svld1_u8(svptrue_b8(), data + 32);
}

/**
 * @brief Perform a vtable lookup in a 16x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint8 vtable_lookup_32bit(
	const vtable8_16x8& tbl,
	vint8 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	svint32_8_t idx_masked = svorr_s32_x(svptrue_b32(), idx.m, svdup_s32(0xFFFFFF00));
	svuint8_8_t idx_bytes = svreinterpret_u8_s32(idx_masked);

	svuint8_8_t result = svtbl_u8(tbl.t0, idx_bytes);
	return vint8(svreinterpret_s32_u8(result));
}

/**
 * @brief Perform a vtable lookup in a 32x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint8 vtable_lookup_32bit(
	const vtable8_32x8& tbl,
	vint8 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	svint32_8_t idx_masked = svorr_s32_x(svptrue_b32(), idx.m, svdup_s32(0xFFFFFF00));
	svuint8_8_t idx_bytes = svreinterpret_u8_s32(idx_masked);

	svuint8_8_t result = svtbl_u8(tbl.t0, idx_bytes);
	return vint8(svreinterpret_s32_u8(result));
}

/**
 * @brief Perform a vtable lookup in a 64x 8-bit table with 32-bit indices.
 *
 * Future: SVE2 can directly do svtbl2_u8() for a two register table.
 */
ASTCENC_SIMD_INLINE vint8 vtable_lookup_32bit(
	const vtable8_64x8& tbl,
	vint8 idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	svint32_8_t idxm = svorr_s32_x(svptrue_b32(), idx.m, svdup_s32(0xFFFFFF00));

	svuint8_8_t idxm8 = svreinterpret_u8_s32(idxm);
	svuint8_8_t t0_lookup = svtbl_u8(tbl.t0, idxm8);

	idxm8 = svsub_u8_x(svptrue_b8(), idxm8, svdup_u8(32));
	svuint8_8_t t1_lookup = svtbl_u8(tbl.t1, idxm8);

	svuint8_8_t result = svorr_u8_x(svptrue_b32(), t0_lookup, t1_lookup);
	return vint8(svreinterpret_s32_u8(result));
}

/**
 * @brief Return a vector of interleaved RGBA data.
 *
 * Input vectors have the value stored in the bottom 8 bits of each lane,
 * with high bits set to zero.
 *
 * Output vector stores a single RGBA texel packed in each lane.
 */
ASTCENC_SIMD_INLINE vint8 interleave_rgba8(vint8 r, vint8 g, vint8 b, vint8 a)
{
	return r + lsl<8>(g) + lsl<16>(b) + lsl<24>(a);
}

/**
 * @brief Store a vector, skipping masked lanes.
 *
 * All masked lanes must be at the end of vector, after all non-masked lanes.
 */
ASTCENC_SIMD_INLINE void store_lanes_masked(uint8_t* base, vint8 data, vmask8 mask)
{
	svst1_s32(mask.m, reinterpret_cast<int32_t*>(base), data.m);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void print(vint8 a)
{
	alignas(32) int v[8];
	storea(a, v);
	printf("v8_i32:\n  %8d %8d %8d %8d %8d %8d %8d %8d\n",
	       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void printx(vint8 a)
{
	alignas(32) int v[8];
	storea(a, v);
	printf("v8_i32:\n  %08x %08x %08x %08x %08x %08x %08x %08x\n",
	       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/**
 * @brief Debug function to print a vector of floats.
 */
ASTCENC_SIMD_INLINE void print(vfloat8 a)
{
	alignas(32) float v[8];
	storea(a, v);
	printf("v8_f32:\n  %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n",
	       static_cast<double>(v[0]), static_cast<double>(v[1]),
	       static_cast<double>(v[2]), static_cast<double>(v[3]),
	       static_cast<double>(v[4]), static_cast<double>(v[5]),
	       static_cast<double>(v[6]), static_cast<double>(v[7]));
}

/**
 * @brief Debug function to print a vector of masks.
 */
ASTCENC_SIMD_INLINE void print(vmask8 a)
{
	print(select(vint8(0), vint8(1), a));
}

#endif // #ifndef ASTC_VECMATHLIB_SVE_8_H_INCLUDED

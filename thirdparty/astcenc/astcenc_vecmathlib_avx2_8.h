// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2019-2022 Arm Limited
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
 * @brief 8x32-bit vectors, implemented using AVX2.
 *
 * This module implements 8-wide 32-bit float, int, and mask vectors for x86
 * AVX2.
 *
 * There is a baseline level of functionality provided by all vector widths and
 * implementations. This is implemented using identical function signatures,
 * modulo data type, so we can use them as substitutable implementations in VLA
 * code.
 */

#ifndef ASTC_VECMATHLIB_AVX2_8_H_INCLUDED
#define ASTC_VECMATHLIB_AVX2_8_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

// Define convenience intrinsics that are missing on older compilers
#define astcenc_mm256_set_m128i(m, n) _mm256_insertf128_si256(_mm256_castsi128_si256((n)), (m), 1)

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
	 * @brief Construct from 4 values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(const float *p)
	{
		m = _mm256_loadu_ps(p);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(float a)
	{
		m = _mm256_set1_ps(a);
	}

	/**
	 * @brief Construct from 8 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(
		float a, float b, float c, float d,
		float e, float f, float g, float h)
	{
		m = _mm256_set_ps(h, g, f, e, d, c, b, a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat8(__m256 a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar value of a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE float lane() const
	{
	#if !defined(__clang__) && defined(_MSC_VER)
		return m.m256_f32[l];
	#else
		union { __m256 m; float f[8]; } cvt;
		cvt.m = m;
		return cvt.f[l];
	#endif
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 zero()
	{
		return vfloat8(_mm256_setzero_ps());
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 load1(const float* p)
	{
		return vfloat8(_mm256_broadcast_ss(p));
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 loada(const float* p)
	{
		return vfloat8(_mm256_load_ps(p));
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vfloat8 lane_id()
	{
		return vfloat8(_mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0));
	}

	/**
	 * @brief The vector ...
	 */
	__m256 m;
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
		m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
	}

	/**
	 * @brief Construct from 8 uint8_t loaded from an unaligned address.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(const uint8_t *p)
	{
		// _mm_loadu_si64 would be nicer syntax, but missing on older GCC
		m = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(*reinterpret_cast<const long long*>(p)));
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using vfloat4::zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(int a)
	{
		m = _mm256_set1_epi32(a);
	}

	/**
	 * @brief Construct from 8 scalar values.
	 *
	 * The value of @c a is stored to lane 0 (LSB) in the SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(
		int a, int b, int c, int d,
		int e, int f, int g, int h)
	{
		m = _mm256_set_epi32(h, g, f, e, d, c, b, a);
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint8(__m256i a)
	{
		m = a;
	}

	/**
	 * @brief Get the scalar from a single lane.
	 */
	template <int l> ASTCENC_SIMD_INLINE int lane() const
	{
	#if !defined(__clang__) && defined(_MSC_VER)
		return m.m256i_i32[l];
	#else
		union { __m256i m; int f[8]; } cvt;
		cvt.m = m;
		return cvt.f[l];
	#endif
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vint8 zero()
	{
		return vint8(_mm256_setzero_si256());
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vint8 load1(const int* p)
	{
		__m128i a = _mm_set1_epi32(*p);
		return vint8(_mm256_broadcastd_epi32(a));
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint8 loada(const int* p)
	{
		return vint8(_mm256_load_si256(reinterpret_cast<const __m256i*>(p)));
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vint8 lane_id()
	{
		return vint8(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
	}

	/**
	 * @brief The vector ...
	 */
	__m256i m;
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
	ASTCENC_SIMD_INLINE explicit vmask8(__m256 a)
	{
		m = a;
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask8(__m256i a)
	{
		m = _mm256_castsi256_ps(a);
	}

	/**
	 * @brief Construct from 1 scalar value.
	 */
	ASTCENC_SIMD_INLINE explicit vmask8(bool a)
	{
		vint8 mask(a == false ? 0 : -1);
		m = _mm256_castsi256_ps(mask.m);
	}

	/**
	 * @brief The vector ...
	 */
	__m256 m;
};

// ============================================================================
// vmask8 operators and functions
// ============================================================================

/**
 * @brief Overload: mask union (or).
 */
ASTCENC_SIMD_INLINE vmask8 operator|(vmask8 a, vmask8 b)
{
	return vmask8(_mm256_or_ps(a.m, b.m));
}

/**
 * @brief Overload: mask intersect (and).
 */
ASTCENC_SIMD_INLINE vmask8 operator&(vmask8 a, vmask8 b)
{
	return vmask8(_mm256_and_ps(a.m, b.m));
}

/**
 * @brief Overload: mask difference (xor).
 */
ASTCENC_SIMD_INLINE vmask8 operator^(vmask8 a, vmask8 b)
{
	return vmask8(_mm256_xor_ps(a.m, b.m));
}

/**
 * @brief Overload: mask invert (not).
 */
ASTCENC_SIMD_INLINE vmask8 operator~(vmask8 a)
{
	return vmask8(_mm256_xor_si256(_mm256_castps_si256(a.m), _mm256_set1_epi32(-1)));
}

/**
 * @brief Return a 8-bit mask code indicating mask status.
 *
 * bit0 = lane 0
 */
ASTCENC_SIMD_INLINE unsigned int mask(vmask8 a)
{
	return static_cast<unsigned int>(_mm256_movemask_ps(a.m));
}

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask8 a)
{
	return mask(a) != 0;
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask8 a)
{
	return mask(a) == 0xFF;
}

// ============================================================================
// vint8 operators and functions
// ============================================================================
/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vint8 operator+(vint8 a, vint8 b)
{
	return vint8(_mm256_add_epi32(a.m, b.m));
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
	return vint8(_mm256_sub_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vint8 operator*(vint8 a, vint8 b)
{
	return vint8(_mm256_mullo_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector bit invert.
 */
ASTCENC_SIMD_INLINE vint8 operator~(vint8 a)
{
	return vint8(_mm256_xor_si256(a.m, _mm256_set1_epi32(-1)));
}

/**
 * @brief Overload: vector by vector bitwise or.
 */
ASTCENC_SIMD_INLINE vint8 operator|(vint8 a, vint8 b)
{
	return vint8(_mm256_or_si256(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise and.
 */
ASTCENC_SIMD_INLINE vint8 operator&(vint8 a, vint8 b)
{
	return vint8(_mm256_and_si256(a.m, b.m));
}

/**
 * @brief Overload: vector by vector bitwise xor.
 */
ASTCENC_SIMD_INLINE vint8 operator^(vint8 a, vint8 b)
{
	return vint8(_mm256_xor_si256(a.m, b.m));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask8 operator==(vint8 a, vint8 b)
{
	return vmask8(_mm256_cmpeq_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask8 operator!=(vint8 a, vint8 b)
{
	return ~vmask8(_mm256_cmpeq_epi32(a.m, b.m));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask8 operator<(vint8 a, vint8 b)
{
	return vmask8(_mm256_cmpgt_epi32(b.m, a.m));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask8 operator>(vint8 a, vint8 b)
{
	return vmask8(_mm256_cmpgt_epi32(a.m, b.m));
}

/**
 * @brief Logical shift left.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 lsl(vint8 a)
{
	return vint8(_mm256_slli_epi32(a.m, s));
}

/**
 * @brief Arithmetic shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 asr(vint8 a)
{
	return vint8(_mm256_srai_epi32(a.m, s));
}

/**
 * @brief Logical shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint8 lsr(vint8 a)
{
	return vint8(_mm256_srli_epi32(a.m, s));
}

/**
 * @brief Return the min vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint8 min(vint8 a, vint8 b)
{
	return vint8(_mm256_min_epi32(a.m, b.m));
}

/**
 * @brief Return the max vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint8 max(vint8 a, vint8 b)
{
	return vint8(_mm256_max_epi32(a.m, b.m));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vint8 hmin(vint8 a)
{
	__m128i m = _mm_min_epi32(_mm256_extracti128_si256(a.m, 0), _mm256_extracti128_si256(a.m, 1));
	m = _mm_min_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,3,2)));
	m = _mm_min_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,1)));
	m = _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,0));

	__m256i r = astcenc_mm256_set_m128i(m, m);
	vint8 vmin(r);
	return vmin;
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vint8 hmax(vint8 a)
{
	__m128i m = _mm_max_epi32(_mm256_extracti128_si256(a.m, 0), _mm256_extracti128_si256(a.m, 1));
	m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,3,2)));
	m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,1)));
	m = _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,0));

	__m256i r = astcenc_mm256_set_m128i(m, m);
	vint8 vmax(r);
	return vmax;
}

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vint8 a, int* p)
{
	_mm256_store_si256(reinterpret_cast<__m256i*>(p), a.m);
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint8 a, int* p)
{
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a.m);
}

/**
 * @brief Store lowest N (vector width) bytes into an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_nbytes(vint8 a, uint8_t* p)
{
	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	// _mm_storeu_si64(ptr, _mm256_extracti128_si256(v.m, 0))
	_mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm256_extracti128_si256(a.m, 0));
}

/**
 * @brief Gather N (vector width) indices from the array.
 */
ASTCENC_SIMD_INLINE vint8 gatheri(const int* base, vint8 indices)
{
	return vint8(_mm256_i32gather_epi32(base, indices.m, 4));
}

/**
 * @brief Pack low 8 bits of N (vector width) lanes into bottom of vector.
 */
ASTCENC_SIMD_INLINE vint8 pack_low_bytes(vint8 v)
{
	__m256i shuf = _mm256_set_epi8(0, 0, 0, 0,  0,  0,  0,  0,
	                               0, 0, 0, 0, 28, 24, 20, 16,
	                               0, 0, 0, 0,  0,  0,  0,  0,
	                               0, 0, 0, 0, 12,  8,  4,  0);
	__m256i a = _mm256_shuffle_epi8(v.m, shuf);
	__m128i a0 = _mm256_extracti128_si256(a, 0);
	__m128i a1 = _mm256_extracti128_si256(a, 1);
	__m128i b = _mm_unpacklo_epi32(a0, a1);

	__m256i r = astcenc_mm256_set_m128i(b, b);
	return vint8(r);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vint8 select(vint8 a, vint8 b, vmask8 cond)
{
	__m256i condi = _mm256_castps_si256(cond.m);
	return vint8(_mm256_blendv_epi8(a.m, b.m, condi));
}

// ============================================================================
// vfloat4 operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vfloat8 operator+(vfloat8 a, vfloat8 b)
{
	return vfloat8(_mm256_add_ps(a.m, b.m));
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
	return vfloat8(_mm256_sub_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(vfloat8 a, vfloat8 b)
{
	return vfloat8(_mm256_mul_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(vfloat8 a, float b)
{
	return vfloat8(_mm256_mul_ps(a.m, _mm256_set1_ps(b)));
}

/**
 * @brief Overload: scalar by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat8 operator*(float a, vfloat8 b)
{
	return vfloat8(_mm256_mul_ps(_mm256_set1_ps(a), b.m));
}

/**
 * @brief Overload: vector by vector division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(vfloat8 a, vfloat8 b)
{
	return vfloat8(_mm256_div_ps(a.m, b.m));
}

/**
 * @brief Overload: vector by scalar division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(vfloat8 a, float b)
{
	return vfloat8(_mm256_div_ps(a.m, _mm256_set1_ps(b)));
}


/**
 * @brief Overload: scalar by vector division.
 */
ASTCENC_SIMD_INLINE vfloat8 operator/(float a, vfloat8 b)
{
	return vfloat8(_mm256_div_ps(_mm256_set1_ps(a), b.m));
}


/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask8 operator==(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_EQ_OQ));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask8 operator!=(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_NEQ_OQ));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask8 operator<(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_LT_OQ));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask8 operator>(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_GT_OQ));
}

/**
 * @brief Overload: vector by vector less than or equal.
 */
ASTCENC_SIMD_INLINE vmask8 operator<=(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_LE_OQ));
}

/**
 * @brief Overload: vector by vector greater than or equal.
 */
ASTCENC_SIMD_INLINE vmask8 operator>=(vfloat8 a, vfloat8 b)
{
	return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_GE_OQ));
}

/**
 * @brief Return the min vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 min(vfloat8 a, vfloat8 b)
{
	return vfloat8(_mm256_min_ps(a.m, b.m));
}

/**
 * @brief Return the min vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 min(vfloat8 a, float b)
{
	return min(a, vfloat8(b));
}

/**
 * @brief Return the max vector of two vectors.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 max(vfloat8 a, vfloat8 b)
{
	return vfloat8(_mm256_max_ps(a.m, b.m));
}

/**
 * @brief Return the max vector of a vector and a scalar.
 *
 * If either lane value is NaN, @c b will be returned for that lane.
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
ASTCENC_SIMD_INLINE vfloat8 clamp(float min, float max, vfloat8 a)
{
	// Do not reorder - second operand will return if either is NaN
	a.m = _mm256_max_ps(a.m, _mm256_set1_ps(min));
	a.m = _mm256_min_ps(a.m, _mm256_set1_ps(max));
	return a;
}

/**
 * @brief Return a clamped value between 0.0f and max.
 *
 * It is assumed that @c max is not a NaN value. If @c a is NaN then zero will
 * be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 clampz(float max, vfloat8 a)
{
	a.m = _mm256_max_ps(a.m, _mm256_setzero_ps());
	a.m = _mm256_min_ps(a.m, _mm256_set1_ps(max));
	return a;
}

/**
 * @brief Return a clamped value between 0.0f and 1.0f.
 *
 * If @c a is NaN then zero will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat8 clampzo(vfloat8 a)
{
	a.m = _mm256_max_ps(a.m, _mm256_setzero_ps());
	a.m = _mm256_min_ps(a.m, _mm256_set1_ps(1.0f));
	return a;
}

/**
 * @brief Return the absolute value of the float vector.
 */
ASTCENC_SIMD_INLINE vfloat8 abs(vfloat8 a)
{
	__m256 msk = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	return vfloat8(_mm256_and_ps(a.m, msk));
}

/**
 * @brief Return a float rounded to the nearest integer value.
 */
ASTCENC_SIMD_INLINE vfloat8 round(vfloat8 a)
{
	constexpr int flags = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
	return vfloat8(_mm256_round_ps(a.m, flags));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat8 hmin(vfloat8 a)
{
	__m128 vlow = _mm256_castps256_ps128(a.m);
	__m128 vhigh = _mm256_extractf128_ps(a.m, 1);
	vlow = _mm_min_ps(vlow, vhigh);

	// First do an horizontal reduction.
	__m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 mins = _mm_min_ps(vlow, shuf);
	shuf = _mm_movehl_ps(shuf, mins);
	mins = _mm_min_ss(mins, shuf);

	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	//__m256i r = _mm256_set_m128(m, m)
	__m256 r = _mm256_insertf128_ps(_mm256_castps128_ps256(mins), mins, 1);

	return vfloat8(_mm256_permute_ps(r, 0));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE float hmin_s(vfloat8 a)
{
	return hmin(a).lane<0>();
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat8 hmax(vfloat8 a)
{
	__m128 vlow = _mm256_castps256_ps128(a.m);
	__m128 vhigh = _mm256_extractf128_ps(a.m, 1);
	vhigh = _mm_max_ps(vlow, vhigh);

	// First do an horizontal reduction.
	__m128 shuf = _mm_shuffle_ps(vhigh, vhigh, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 maxs = _mm_max_ps(vhigh, shuf);
	shuf = _mm_movehl_ps(shuf,maxs);
	maxs = _mm_max_ss(maxs, shuf);

	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	//__m256i r = _mm256_set_m128(m, m)
	__m256 r = _mm256_insertf128_ps(_mm256_castps128_ps256(maxs), maxs, 1);
	return vfloat8(_mm256_permute_ps(r, 0));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE float hmax_s(vfloat8 a)
{
	return hmax(a).lane<0>();
}

/**
 * @brief Return the horizontal sum of a vector.
 */
ASTCENC_SIMD_INLINE float hadd_s(vfloat8 a)
{
	// Two sequential 4-wide adds gives invariance with 4-wide code
	vfloat4 lo(_mm256_extractf128_ps(a.m, 0));
	vfloat4 hi(_mm256_extractf128_ps(a.m, 1));
	return hadd_s(lo) + hadd_s(hi);
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat8 select(vfloat8 a, vfloat8 b, vmask8 cond)
{
	return vfloat8(_mm256_blendv_ps(a.m, b.m, cond.m));
}

/**
 * @brief Return lanes from @c b if MSB of @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat8 select_msb(vfloat8 a, vfloat8 b, vmask8 cond)
{
	return vfloat8(_mm256_blendv_ps(a.m, b.m, cond.m));
}

/**
 * @brief Accumulate lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat8 a)
{
	vfloat4 lo(_mm256_extractf128_ps(a.m, 0));
	haccumulate(accum, lo);

	vfloat4 hi(_mm256_extractf128_ps(a.m, 1));
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
	a = select(vfloat8::zero(), a, m);
	haccumulate(accum, a);
}

/**
 * @brief Return the sqrt of the lanes in the vector.
 */
ASTCENC_SIMD_INLINE vfloat8 sqrt(vfloat8 a)
{
	return vfloat8(_mm256_sqrt_ps(a.m));
}

/**
 * @brief Load a vector of gathered results from an array;
 */
ASTCENC_SIMD_INLINE vfloat8 gatherf(const float* base, vint8 indices)
{
	return vfloat8(_mm256_i32gather_ps(base, indices.m, 4));
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vfloat8 a, float* p)
{
	_mm256_storeu_ps(p, a.m);
}

/**
 * @brief Store a vector to a 32B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vfloat8 a, float* p)
{
	_mm256_store_ps(p, a.m);
}

/**
 * @brief Return a integer value for a float vector, using truncation.
 */
ASTCENC_SIMD_INLINE vint8 float_to_int(vfloat8 a)
{
	return vint8(_mm256_cvttps_epi32(a.m));
}

/**
 * @brief Return a integer value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint8 float_to_int_rtn(vfloat8 a)
{
	a = round(a);
	return vint8(_mm256_cvttps_epi32(a.m));
}


/**
 * @brief Return a float value for an integer vector.
 */
ASTCENC_SIMD_INLINE vfloat8 int_to_float(vint8 a)
{
	return vfloat8(_mm256_cvtepi32_ps(a.m));
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
	return vint8(_mm256_castps_si256(a.m));
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
	return vfloat8(_mm256_castsi256_ps(a.m));
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(vint4 t0, vint8& t0p)
{
	// AVX2 duplicates the table within each 128-bit lane
	__m128i t0n = t0.m;
	t0p = vint8(astcenc_mm256_set_m128i(t0n, t0n));
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(vint4 t0, vint4 t1, vint8& t0p, vint8& t1p)
{
	// AVX2 duplicates the table within each 128-bit lane
	__m128i t0n = t0.m;
	t0p = vint8(astcenc_mm256_set_m128i(t0n, t0n));

	__m128i t1n = _mm_xor_si128(t0.m, t1.m);
	t1p = vint8(astcenc_mm256_set_m128i(t1n, t1n));
}

/**
 * @brief Prepare a vtable lookup table for use with the native SIMD size.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vint4 t0, vint4 t1, vint4 t2, vint4 t3,
	vint8& t0p, vint8& t1p, vint8& t2p, vint8& t3p)
{
	// AVX2 duplicates the table within each 128-bit lane
	__m128i t0n = t0.m;
	t0p = vint8(astcenc_mm256_set_m128i(t0n, t0n));

	__m128i t1n = _mm_xor_si128(t0.m, t1.m);
	t1p = vint8(astcenc_mm256_set_m128i(t1n, t1n));

	__m128i t2n = _mm_xor_si128(t1.m, t2.m);
	t2p = vint8(astcenc_mm256_set_m128i(t2n, t2n));

	__m128i t3n = _mm_xor_si128(t2.m, t3.m);
	t3p = vint8(astcenc_mm256_set_m128i(t3n, t3n));
}

/**
 * @brief Perform an 8-bit 16-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint8 vtable_8bt_32bi(vint8 t0, vint8 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m256i idxx = _mm256_or_si256(idx.m, _mm256_set1_epi32(static_cast<int>(0xFFFFFF00)));

	__m256i result = _mm256_shuffle_epi8(t0.m, idxx);
	return vint8(result);
}

/**
 * @brief Perform an 8-bit 32-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint8 vtable_8bt_32bi(vint8 t0, vint8 t1, vint8 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m256i idxx = _mm256_or_si256(idx.m, _mm256_set1_epi32(static_cast<int>(0xFFFFFF00)));

	__m256i result = _mm256_shuffle_epi8(t0.m, idxx);
	idxx = _mm256_sub_epi8(idxx, _mm256_set1_epi8(16));

	__m256i result2 = _mm256_shuffle_epi8(t1.m, idxx);
	result = _mm256_xor_si256(result, result2);
	return vint8(result);
}

/**
 * @brief Perform an 8-bit 64-entry table lookup, with 32-bit indexes.
 */
ASTCENC_SIMD_INLINE vint8 vtable_8bt_32bi(vint8 t0, vint8 t1, vint8 t2, vint8 t3, vint8 idx)
{
	// Set index byte MSB to 1 for unused bytes so shuffle returns zero
	__m256i idxx = _mm256_or_si256(idx.m, _mm256_set1_epi32(static_cast<int>(0xFFFFFF00)));

	__m256i result = _mm256_shuffle_epi8(t0.m, idxx);
	idxx = _mm256_sub_epi8(idxx, _mm256_set1_epi8(16));

	__m256i result2 = _mm256_shuffle_epi8(t1.m, idxx);
	result = _mm256_xor_si256(result, result2);
	idxx = _mm256_sub_epi8(idxx, _mm256_set1_epi8(16));

	result2 = _mm256_shuffle_epi8(t2.m, idxx);
	result = _mm256_xor_si256(result, result2);
	idxx = _mm256_sub_epi8(idxx, _mm256_set1_epi8(16));

	result2 = _mm256_shuffle_epi8(t3.m, idxx);
	result = _mm256_xor_si256(result, result2);

	return vint8(result);
}

/**
 * @brief Return a vector of interleaved RGBA data.
 *
 * Input vectors have the value stored in the bottom 8 bits of each lane,
 * with high  bits set to zero.
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
ASTCENC_SIMD_INLINE void store_lanes_masked(int* base, vint8 data, vmask8 mask)
{
	_mm256_maskstore_epi32(base, _mm256_castps_si256(mask.m), data.m);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void print(vint8 a)
{
	alignas(ASTCENC_VECALIGN) int v[8];
	storea(a, v);
	printf("v8_i32:\n  %8d %8d %8d %8d %8d %8d %8d %8d\n",
	       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void printx(vint8 a)
{
	alignas(ASTCENC_VECALIGN) int v[8];
	storea(a, v);
	printf("v8_i32:\n  %08x %08x %08x %08x %08x %08x %08x %08x\n",
	       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/**
 * @brief Debug function to print a vector of floats.
 */
ASTCENC_SIMD_INLINE void print(vfloat8 a)
{
	alignas(ASTCENC_VECALIGN) float v[8];
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

#endif // #ifndef ASTC_VECMATHLIB_AVX2_8_H_INCLUDED

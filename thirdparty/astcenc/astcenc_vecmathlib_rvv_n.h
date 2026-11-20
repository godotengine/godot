// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2026 Arm Limited
// Copyright 2026 Olaf Bernstein
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
 * @brief Nx32-bit vectors, implemented using RVV.
 *
 * This module implements N-wide 32-bit float, int, and mask vectors for the
 * RISC-V "V" extension.
 *
 * There is a baseline level of functionality provided by all vector widths and
 * implementations. This is implemented using identical function signatures,
 * modulo data type, so we can use them as substitutable implementations in VLA
 * code.
 */

#ifndef ASTC_VECMATHLIB_RVV_N_H_INCLUDED
#define ASTC_VECMATHLIB_RVV_N_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

#define ASTCENC_SIMD_WIDTH  (__riscv_v_fixed_vlen/32)

typedef vbool32_t    vbool_t   __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH)));
typedef vuint8m1_t   vuint8_t  __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH * 32)));
typedef vuint16m1_t  vuint16_t __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH * 32)));
typedef vuint32m1_t  vuint32_t __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH * 32)));
typedef vint32m1_t   vint32_t  __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH * 32)));
typedef vfloat32m1_t vfloat_t  __attribute__((riscv_rvv_vector_bits(ASTCENC_SIMD_WIDTH * 32)));

// ============================================================================
// vfloat data type
// ============================================================================

/**
 * @brief Data type for N-wide floats.
 */
struct vfloat
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vfloat() = default;

	/**
	 * @brief Construct from N values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat(const float *p)
	{
		m = __riscv_vle32_v_f32m1(p, vl());
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat(float a)
	{
		m = __riscv_vfmv_v_f_f32m1(a, vl());
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vfloat(vfloat_t a)
	{
		m = a;
	}

	/**
	 * @brief Return vector length of type.
	 */
	static ASTCENC_SIMD_INLINE size_t vl()
	{
		return __riscv_vsetvlmax_e32m1();
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vfloat zero()
	{
		return vfloat(0.0f);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat load1(const float* p)
	{
		return vfloat(*p);
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vfloat loada(const float* p)
	{
		return vfloat(p);
	}

	/**
	 * @brief The vector ...
	 */
	vfloat_t m;
};

// ============================================================================
// vint data type
// ============================================================================

/**
 * @brief Data type for N-wide ints.
 */
struct vint
{
	/**
	 * @brief Construct from zero-initialized value.
	 */
	ASTCENC_SIMD_INLINE vint() = default;

	/**
	 * @brief Construct from N values loaded from an unaligned address.
	 *
	 * Consider using loada() which is better with vectors if data is aligned
	 * to vector length.
	 */
	ASTCENC_SIMD_INLINE explicit vint(const int *p)
	{
		m = __riscv_vle32_v_i32m1(p, vl());
	}

	/**
	 * @brief Construct from N uint8_t loaded from an unaligned address.
	 */
	ASTCENC_SIMD_INLINE explicit vint(const uint8_t *p)
	{
		// Load 8-bit values and expand to 32-bits
		m = __riscv_vreinterpret_i32m1(
				__riscv_vzext_vf4(
					__riscv_vle8_v_u8mf4(p, vl()), vl()
				)
			);
	}

	/**
	 * @brief Construct from 1 scalar value replicated across all lanes.
	 *
	 * Consider using zero() for constexpr zeros.
	 */
	ASTCENC_SIMD_INLINE explicit vint(int a)
	{
		m = __riscv_vmv_v_x_i32m1(a, vl());
	}

	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vint(vint32_t a)
	{
		m = a;
	}

	/**
	 * @brief Return vector length of type.
	 */
	static ASTCENC_SIMD_INLINE size_t vl()
	{
		return __riscv_vsetvlmax_e32m1();
	}

	/**
	 * @brief Factory that returns a vector of zeros.
	 */
	static ASTCENC_SIMD_INLINE vint zero()
	{
		return vint(0.0f);
	}

	/**
	 * @brief Factory that returns a replicated scalar loaded from memory.
	 */
	static ASTCENC_SIMD_INLINE vint load1(const int* p)
	{
		return vint(*p);
	}

	/**
	 * @brief Factory that returns a vector loaded from unaligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint load(const uint8_t* p)
	{
		vuint8_t v = __riscv_vle8_v_u8m1(p, __riscv_vsetvlmax_e8m1());
		return vint(__riscv_vreinterpret_i32m1(__riscv_vreinterpret_u32m1(v)));
	}

	/**
	 * @brief Factory that returns a vector loaded from 32B aligned memory.
	 */
	static ASTCENC_SIMD_INLINE vint loada(const int* p)
	{
		return vint(p);
	}

	/**
	 * @brief Factory that returns a vector containing the lane IDs.
	 */
	static ASTCENC_SIMD_INLINE vint lane_id()
	{
		return vint(__riscv_vreinterpret_i32m1(__riscv_vid_v_u32m1(vl())));
	}

	/**
	 * @brief The vector ...
	 */
	 vint32_t m;
};

// ============================================================================
// vmask data type
// ============================================================================

/**
 * @brief Data type for N-wide control plane masks.
 */
struct vmask
{
	/**
	 * @brief Construct from an existing SIMD register.
	 */
	ASTCENC_SIMD_INLINE explicit vmask(vbool_t a)
	{
		m = a;
	}

	/**
	 * @brief Construct from 1 scalar value.
	 */
	ASTCENC_SIMD_INLINE explicit vmask(bool a)
	{
		m = a ? __riscv_vmset_m_b32(vl()) : __riscv_vmclr_m_b32(vl());
	}

	/**
	 * @brief Return vector length of type.
	 */
	static ASTCENC_SIMD_INLINE size_t vl()
	{
		return __riscv_vsetvlmax_e32m1();
	}

	/**
	 * @brief The vector ...
	 */
	vbool_t m;
};

// ============================================================================
// vmask operators and functions
// ============================================================================

/**
 * @brief Overload: mask union (or).
 */
ASTCENC_SIMD_INLINE vmask operator|(vmask a, vmask b)
{
	return vmask(__riscv_vmor(a.m, b.m, vmask::vl()));
}

/**
 * @brief Overload: mask intersect (and).
 */
ASTCENC_SIMD_INLINE vmask operator&(vmask a, vmask b)
{
	return vmask(__riscv_vmand(a.m, b.m, vmask::vl()));
}

/**
 * @brief Overload: mask difference (xor).
 */
ASTCENC_SIMD_INLINE vmask operator^(vmask a, vmask b)
{
	return vmask(__riscv_vmxor(a.m, b.m, vmask::vl()));
}

/**
 * @brief Overload: mask invert (not).
 */
ASTCENC_SIMD_INLINE vmask operator~(vmask a)
{
	return vmask(__riscv_vmnot(a.m, vmask::vl()));
}

#if ASTCENC_SIMD_WIDTH <= 64
/**
 * @brief Return a N-bit mask code indicating mask status.
 *
 * bit0 = lane 0
 */
ASTCENC_SIMD_INLINE uint64_t mask(vmask a)
{
	return __riscv_vmv_x(__riscv_vreinterpret_u64m1(a.m)) & (~(uint64_t)0 >> (64-vmask::vl()));
}
#else
/* only used for in the current unit-tests */
#endif

/**
 * @brief True if any lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool any(vmask a)
{
	return __riscv_vfirst(a.m, vmask::vl()) >= 0;
}

/**
 * @brief True if all lanes are enabled, false otherwise.
 */
ASTCENC_SIMD_INLINE bool all(vmask a)
{
	return __riscv_vcpop(a.m, vmask::vl()) == vmask::vl();
}

// ============================================================================
// vint operators and functions
// ============================================================================
/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vint operator+(vint a, vint b)
{
	return vint(__riscv_vadd(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vint& operator+=(vint& a, const vint& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vint operator-(vint a, vint b)
{
	return vint(__riscv_vsub(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vint operator*(vint a, vint b)
{
	return vint(__riscv_vmul(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector bit invert.
 */
ASTCENC_SIMD_INLINE vint operator~(vint a)
{
	return vint(__riscv_vnot(a.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector bitwise or.
 */
ASTCENC_SIMD_INLINE vint operator|(vint a, vint b)
{
	return vint(__riscv_vor(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector bitwise and.
 */
ASTCENC_SIMD_INLINE vint operator&(vint a, vint b)
{
	return vint(__riscv_vand(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector bitwise xor.
 */
ASTCENC_SIMD_INLINE vint operator^(vint a, vint b)
{
	return vint(__riscv_vxor(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector equality.
 */
ASTCENC_SIMD_INLINE vmask operator==(vint a, vint b)
{
	return vmask(__riscv_vmseq(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector inequality.
 */
ASTCENC_SIMD_INLINE vmask operator!=(vint a, vint b)
{
	return vmask(__riscv_vmsne(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask operator<(vint a, vint b)
{
	return vmask(__riscv_vmslt(a.m, b.m, vint::vl()));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask operator>(vint a, vint b)
{
	return vmask(__riscv_vmsgt(a.m, b.m, vint::vl()));
}

/**
 * @brief Logical shift left.
 */
template <int s> ASTCENC_SIMD_INLINE vint lsl(vint a)
{
	return vint(__riscv_vsll(a.m, s, vint::vl()));
}

/**
 * @brief Arithmetic shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint asr(vint a)
{
	return vint(__riscv_vsra(a.m, s, vint::vl()));
}

/**
 * @brief Logical shift right.
 */
template <int s> ASTCENC_SIMD_INLINE vint lsr(vint a)
{
	vuint32_t r = __riscv_vreinterpret_u32m1(a.m);
	r = __riscv_vsrl(r, s, vint::vl());
	return vint(__riscv_vreinterpret_i32m1(r));
}

/**
 * @brief Return the min vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint min(vint a, vint b)
{
	return vint(__riscv_vmin(a.m, b.m, vint::vl()));
}

/**
 * @brief Return the max vector of two vectors.
 */
ASTCENC_SIMD_INLINE vint max(vint a, vint b)
{
	return vint(__riscv_vmax(a.m, b.m, vint::vl()));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vint hmin(vint a)
{
	a.m = __riscv_vredmin(a.m, a.m, vint::vl());
	return vint(__riscv_vrgather(a.m, 0, vint::vl()));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE int hmin_s(vint a)
{
	return __riscv_vmv_x(__riscv_vredmin(a.m, a.m, vint::vl()));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vint hmax(vint a)
{
	a.m = __riscv_vredmax(a.m, a.m, vint::vl());
	return vint(__riscv_vrgather(a.m, 0, vint::vl()));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE int hmax_s(vint a)
{
	return __riscv_vmv_x(__riscv_vredmax(a.m, a.m, vint::vl()));
}

/**
 * @brief Generate a vint from a size_t.
 */
 ASTCENC_SIMD_INLINE vint vint_from_size(size_t a)
 {
	assert(a <= std::numeric_limits<int>::max());
	return vint(static_cast<int>(a));
 }

/**
 * @brief Store a vector to a 16B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vint a, int* p)
{
	__riscv_vse32(p, a.m, vint::vl());
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vint a, int* p)
{
	__riscv_vse32(p, a.m, vint::vl());
}

/**
 * @brief Store lowest N (vector width) bytes into an unaligned address.
 */
ASTCENC_SIMD_INLINE void store_nbytes(vint a, uint8_t* p)
{
	__riscv_vse8((int8_t*)p, __riscv_vreinterpret_i8m1(a.m), vint::vl());
}

/**
 * @brief Pack low 8 bits of N (vector width) lanes into bottom of vector.
 */
ASTCENC_SIMD_INLINE void pack_and_store_low_bytes(vint v, uint8_t* p)
{
	vuint32_t r = __riscv_vreinterpret_u32m1(v.m);
	__riscv_vse8(p, __riscv_vncvt_x(__riscv_vncvt_x(r, vint::vl()), vint::vl()), vint::vl());
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vint select(vint a, vint b, vmask cond)
{
	return vint(__riscv_vmerge(a.m, b.m, cond.m, vint::vl()));
}

// ============================================================================
// vfloat operators and functions
// ============================================================================

/**
 * @brief Overload: vector by vector addition.
 */
ASTCENC_SIMD_INLINE vfloat operator+(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfadd(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector incremental addition.
 */
ASTCENC_SIMD_INLINE vfloat& operator+=(vfloat& a, const vfloat& b)
{
	a = a + b;
	return a;
}

/**
 * @brief Overload: vector by vector subtraction.
 */
ASTCENC_SIMD_INLINE vfloat operator-(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfsub(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat operator*(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfmul(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by scalar multiplication.
 */
ASTCENC_SIMD_INLINE vfloat operator*(vfloat a, float b)
{
	return vfloat(__riscv_vfmul(a.m, b, vfloat::vl()));
}

/**
 * @brief Overload: scalar by vector multiplication.
 */
ASTCENC_SIMD_INLINE vfloat operator*(float a, vfloat b)
{
	return vfloat(__riscv_vfmul(b.m, a, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector division.
 */
ASTCENC_SIMD_INLINE vfloat operator/(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfdiv(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by scalar division.
 */
ASTCENC_SIMD_INLINE vfloat operator/(vfloat a, float b)
{
	return vfloat(__riscv_vfdiv(a.m, b, vfloat::vl()));
}

/**
 * @brief Overload: scalar by vector division.
 */
ASTCENC_SIMD_INLINE vfloat operator/(float a, vfloat b)
{
	return vfloat(__riscv_vfrdiv(b.m, a, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector equality.
 *
 * Returns vector of false mask values if a or b is NaN.
 */
ASTCENC_SIMD_INLINE vmask operator==(vfloat a, vfloat b)
{
	return vmask(__riscv_vmfeq(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector inequality.
 *
 * Returns vector of true mask values if a or b is NaN.
 */
ASTCENC_SIMD_INLINE vmask operator!=(vfloat a, vfloat b)
{
	return vmask(__riscv_vmfne(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector less than.
 */
ASTCENC_SIMD_INLINE vmask operator<(vfloat a, vfloat b)
{
	return vmask(__riscv_vmflt(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector greater than.
 */
ASTCENC_SIMD_INLINE vmask operator>(vfloat a, vfloat b)
{
	return vmask(__riscv_vmfgt(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector less than or equal.
 */
ASTCENC_SIMD_INLINE vmask operator<=(vfloat a, vfloat b)
{
	return vmask(__riscv_vmfle(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Overload: vector by vector greater than or equal.
 */
ASTCENC_SIMD_INLINE vmask operator>=(vfloat a, vfloat b)
{
	return vmask(__riscv_vmfge(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Return the min vector of two vectors.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat min(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfmin(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Return the min vector of a vector and a scalar.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat min(vfloat a, float b)
{
	return vfloat(__riscv_vfmin(a.m, b, vfloat::vl()));
}

/**
 * @brief Return the max vector of two vectors.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat max(vfloat a, vfloat b)
{
	return vfloat(__riscv_vfmax(a.m, b.m, vfloat::vl()));
}

/**
 * @brief Return the max vector of a vector and a scalar.
 *
 * If either lane value is NaN, the other lane will be returned.
 */
ASTCENC_SIMD_INLINE vfloat max(vfloat a, float b)
{
	return vfloat(__riscv_vfmax(a.m, b, vfloat::vl()));
}

/**
 * @brief Return the clamped value between min and max.
 *
 * It is assumed that neither @c min nor @c max are NaN values. If @c a is NaN
 * then @c min will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat clamp(float minv, float maxv, vfloat a)
{
	return min(max(a, minv), maxv);
}

/**
 * @brief Return a clamped value between 0.0f and 1.0f.
 *
 * If @c a is NaN then zero will be returned for that lane.
 */
ASTCENC_SIMD_INLINE vfloat clampzo(vfloat a)
{
	return clamp(0.0f, 1.0f, a);
}

/**
 * @brief Return the absolute value of the float vector.
 */
ASTCENC_SIMD_INLINE vfloat abs(vfloat a)
{
	return vfloat(__riscv_vfabs(a.m, vfloat::vl()));
}

/**
 * @brief Return a float rounded to the nearest integer value.
 */
ASTCENC_SIMD_INLINE vfloat round(vfloat a)
{
	vint32_t r = __riscv_vfcvt_x(a.m, 0, vfloat::vl());
	return vfloat(__riscv_vfcvt_f(r, 0, vfloat::vl()));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat hmin(vfloat a)
{
	a.m = __riscv_vfredmin(a.m, a.m, vint::vl());
	return vfloat(__riscv_vrgather(a.m, 0, vint::vl()));
}

/**
 * @brief Return the horizontal minimum of a vector.
 */
ASTCENC_SIMD_INLINE float hmin_s(vfloat a)
{
	return __riscv_vfmv_f(__riscv_vfredmin(a.m, a.m, vint::vl()));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE vfloat hmax(vfloat a)
{
	a.m = __riscv_vfredmax(a.m, a.m, vint::vl());
	return vfloat(__riscv_vrgather(a.m, 0, vint::vl()));
}

/**
 * @brief Return the horizontal maximum of a vector.
 */
ASTCENC_SIMD_INLINE float hmax_s(vfloat a)
{
	return __riscv_vfmv_f(__riscv_vfredmax(a.m, a.m, vint::vl()));
}

/**
 * @brief Return the horizontal sum of a vector.
 */
ASTCENC_SIMD_INLINE float hadd_s(vfloat a)
{
#if defined(ASTCENC_NO_INVARIANCE)
	vfloat_t z = __riscv_vfmv_v_f_f32m1(0, vfloat::vl());
	return __riscv_vfmv_f(__riscv_vfredusum(a.m, z, vfloat::vl()));
#else
	// TODO: Zvzip variant, once it's ratified
	size_t vl = __riscv_vsetvlmax_e64m1();
	vuint64m1_t r = __riscv_vreinterpret_u64m1(__riscv_vreinterpret_u32m1(a.m));
	vuint64m1_t r0 = __riscv_vreinterpret_u64m1(__riscv_vlmul_ext_u32m1(__riscv_vnsrl(r, 0, vl)));
	vuint64m1_t r1 = __riscv_vreinterpret_u64m1(__riscv_vlmul_ext_u32m1(__riscv_vnsrl(r, 32, vl)));
	vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0, 1);
	vl /= 2;
	vfloat32mf2_t r00 = __riscv_vreinterpret_f32mf2(__riscv_vnsrl(r0, 0, vl));
	vfloat32mf2_t r01 = __riscv_vreinterpret_f32mf2(__riscv_vnsrl(r0, 32, vl));
	vfloat32mf2_t r10 = __riscv_vreinterpret_f32mf2(__riscv_vnsrl(r1, 0, vl));
	vfloat32mf2_t r11 = __riscv_vreinterpret_f32mf2(__riscv_vnsrl(r1, 32, vl));
	vfloat32mf2_t sum = __riscv_vfadd(__riscv_vfadd(r00, r10, vl), __riscv_vfadd(r01, r11, vl), vl);
	return __riscv_vfmv_f(__riscv_vfredosum(sum, z, vl));
#endif
}

/**
 * @brief Return lanes from @c b if @c cond is set, else @c a.
 */
ASTCENC_SIMD_INLINE vfloat select(vfloat a, vfloat b, vmask cond)
{
	return vfloat(__riscv_vmerge(a.m, b.m, cond.m, vint::vl()));
}

/**
 * @brief Accumulate lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat a)
{
#if defined(ASTCENC_NO_INVARIANCE)
	accum.m[0] += hadd_s(a);
#else
	vfloat_t vacc = __riscv_vle32_v_f32m1(accum.m, 4);
	vacc = __riscv_vfadd(vacc, a.m, 4);
	for (size_t i = 4; i < vfloat::vl(); i += 4)
	{
		vacc = __riscv_vfadd(vacc, __riscv_vslidedown(a.m, i, 4), 4);
	}
	__riscv_vse32(accum.m, vacc, 4);
#endif
}

/**
 * @brief Accumulate lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat& accum, vfloat a)
{
	accum += a;
}

/**
 * @brief Accumulate masked lane-wise sums for a vector, folded 4-wide.
 *
 * This is invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat4& accum, vfloat a, vmask m)
{
	a = select(vfloat::zero(), a, m);
	haccumulate(accum, a);
}

/**
 * @brief Accumulate masked lane-wise sums for a vector.
 *
 * This is NOT invariant with 4-wide implementations.
 */
ASTCENC_SIMD_INLINE void haccumulate(vfloat& accum, vfloat a, vmask m)
{
	accum.m = __riscv_vfadd_mu(m.m, accum.m, accum.m, a.m, vfloat::vl());
}

/**
 * @brief Return the sqrt of the lanes in the vector.
 */
ASTCENC_SIMD_INLINE vfloat sqrt(vfloat a)
{
	return vfloat(__riscv_vfsqrt(a.m, vfloat::vl()));
}

/**
 * @brief Load a vector of gathered results from an array;
 */
ASTCENC_SIMD_INLINE vfloat gatherf(const float* base, vint indices)
{
	// NOTE: currently unused,
	//       breaks for negative indices or ones >=2^30
	vuint32m1_t vidx = __riscv_vreinterpret_u32m1(indices.m);
	vidx = __riscv_vsll(vidx, 2, vfloat::vl());
	return vfloat(__riscv_vluxei32_v_f32m1(base, vidx, vfloat::vl()));
}

/**
 * @brief Load a vector of gathered results from an array using byte indices from memory
 */
template<>
ASTCENC_SIMD_INLINE vfloat gatherf_byte_inds<vfloat>(const float* base, const uint8_t* indices)
{
	vuint16mf2_t vidx = __riscv_vwmulu(
		__riscv_vle8_v_u8mf4(indices, vfloat::vl()),
		4, vfloat::vl());

	return vfloat(__riscv_vluxei16_v_f32m1(base, vidx, vfloat::vl()));
}

/**
 * @brief Store a vector to an unaligned memory address.
 */
ASTCENC_SIMD_INLINE void store(vfloat a, float* p)
{
	__riscv_vse32(p, a.m, vfloat::vl());
}

/**
 * @brief Store a vector to a 32B aligned memory address.
 */
ASTCENC_SIMD_INLINE void storea(vfloat a, float* p)
{
	__riscv_vse32(p, a.m, vfloat::vl());
}

/**
 * @brief Return a integer value for a float vector, using truncation.
 */
ASTCENC_SIMD_INLINE vint float_to_int(vfloat a)
{
	return vint(__riscv_vfcvt_rtz_x(a.m, vfloat::vl()));
}

/**
 * @brief Return a integer value for a float vector, using round-to-nearest.
 */
ASTCENC_SIMD_INLINE vint float_to_int_rtn(vfloat a)
{
	return vint(__riscv_vfcvt_rtz_x(__riscv_vfadd(a.m, 0.5f, vfloat::vl()), vfloat::vl()));
}

/**
 * @brief Return a float value for an integer vector.
 */
ASTCENC_SIMD_INLINE vfloat int_to_float(vint a)
{
	return vfloat(__riscv_vfcvt_f(a.m, vfloat::vl()));
}

/**
 * @brief Return a float value as an integer bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the first half of that flip.
 */
ASTCENC_SIMD_INLINE vint float_as_int(vfloat a)
{
	return vint(__riscv_vreinterpret_i32m1(a.m));
}

/**
 * @brief Return a integer value as a float bit pattern (i.e. no conversion).
 *
 * It is a common trick to convert floats into integer bit patterns, perform
 * some bit hackery based on knowledge they are IEEE 754 layout, and then
 * convert them back again. This is the second half of that flip.
 */
ASTCENC_SIMD_INLINE vfloat int_as_float(vint a)
{
	return vfloat(__riscv_vreinterpret_f32m1(a.m));
}

typedef vuint8m1_t vuint8m1fix_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

/*
 * Table structure for a 16x 8-bit entry table.
 */
struct vtable_16x8 {
	vuint8m1fix_t t0;
};

/*
 * Table structure for a 32x 8-bit entry table.
 */
struct vtable_32x8 {
#if __riscv_v_fixed_vlen >= (32 * 8)
	vuint8_t t0;
#else /* minimum VLEN is 128 */
	vuint8_t t0, t1;
#endif
};

/*
 * Table structure for a 64x 8-bit entry table.
 */
struct vtable_64x8 {
#if __riscv_v_fixed_vlen >= (64 * 8)
	/* currently unreachable and not tested */
	vuint8_t t0;
#elif (__riscv_v_fixed_vlen * 2) >= (64 * 8)
	vuint8_t t0, t1;
#else /* minimum VLEN is 128 */
	vuint8_t t0, t1, t2, t3;
#endif
};

/**
 * @brief Prepare a vtable lookup table for 16x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable_16x8& table,
	const uint8_t* data
) {
	vuint8m1_t z = __riscv_vmv_v_x_u8m1(0, __riscv_vsetvlmax_e8m1());
	table.t0 = __riscv_vle8_tu(z, data, 16);
}

/**
 * @brief Prepare a vtable lookup table for 32x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable_32x8& table,
	const uint8_t* data
) {
#if __riscv_v_fixed_vlen >= (32 * 8)
	vuint8m1_t z = __riscv_vmv_v_x_u8m1(0, __riscv_vsetvlmax_e8m1());
	table.t0 = __riscv_vle8_tu(z, data, 32);
#else /* minimum VLEN is 128 */
	table.t0 = __riscv_vle8_v_u8m1(data + 0 * 16, 16);
	table.t1 = __riscv_vle8_v_u8m1(data + 1 * 16, 16);
#endif
}

/**
 * @brief Prepare a vtable lookup table 64x 8-bit entry table.
 */
ASTCENC_SIMD_INLINE void vtable_prepare(
	vtable_64x8& table,
	const uint8_t* data
) {
#if __riscv_v_fixed_vlen >= (64 * 8)
	/* currently unreachable and not tested */
	vuint8m1_t z = __riscv_vmv_v_x_u8m1(0, __riscv_vsetvlmax_e8m1());
	table.t0 = __riscv_vle8_tu(z, data, 64);
#elif (__riscv_v_fixed_vlen * 2) >= (64 * 8)
	table.t0 = __riscv_vle8_v_u8m1(data+0*32, 32);
	table.t1 = __riscv_vle8_v_u8m1(data+1*32, 32);
#else /* minimum VLEN is 128 */
	table.t0 = __riscv_vle8_v_u8m1(data+0*16, 16);
	table.t1 = __riscv_vle8_v_u8m1(data+1*16, 16);
	table.t2 = __riscv_vle8_v_u8m1(data+2*16, 16);
	table.t3 = __riscv_vle8_v_u8m1(data+3*16, 16);
#endif
}

/**
 * @brief Perform a vtable lookup in a 16x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint vtable_lookup_32bit(
	const vtable_16x8& tbl,
	vint idx
) {
	// Set index byte above max index for unused bytes so table lookup returns zero
	vint32_t idx_masked = __riscv_vor(idx.m, 0xFFFFFF00, vint::vl());
	vuint8_t idx_bytes = __riscv_vreinterpret_u8m1(__riscv_vreinterpret_u32m1(idx_masked));
	size_t vl = __riscv_vsetvlmax_e8m1();
	vuint8_t res = __riscv_vrgather(tbl.t0, idx_bytes, vl);
	return vint(__riscv_vreinterpret_i32m1(__riscv_vreinterpret_u32m1(res)));
}

/**
 * @brief Perform a vtable lookup in a 32x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint vtable_lookup_32bit(
	const vtable_32x8& tbl,
	vint idx
) {
#if __riscv_v_fixed_vlen >= (32 * 8)
	vtable_16x8 tbl16;
	tbl16.t0 = tbl.t0;
	return vtable_lookup_32bit(tbl16, idx);
#else /* minimum VLEN is 128 */
	vint32_t idx_masked = __riscv_vor(idx.m, 0xFFFFFF00, vint::vl());
	size_t vl = __riscv_vsetvlmax_e8m1();
	vuint8_t idx_bytes0 = __riscv_vreinterpret_u8m1(__riscv_vreinterpret_u32m1(idx_masked));
	vuint8_t idx_bytes1 = __riscv_vadd(idx_bytes0, -16, vl);
	vbool8_t lt16 = __riscv_vmsltu(idx_bytes0, 16, vl);
	vuint8_t res = __riscv_vrgather(tbl.t1, idx_bytes1, vl);
	res = __riscv_vrgather_mu(lt16, res, tbl.t0, idx_bytes0, vl);
	return vint(__riscv_vreinterpret_i32m1(__riscv_vreinterpret_u32m1(res)));
#endif
}

/**
 * @brief Perform a vtable lookup in a 64x 8-bit table with 32-bit indices.
 */
ASTCENC_SIMD_INLINE vint vtable_lookup_32bit(
	const vtable_64x8& tbl,
	vint idx
) {
#if __riscv_v_fixed_vlen >= (64 * 8)
	/* currently unreachable and not tested */
	vtable_16x8 tbl16;
	tbl16.t0 = tbl.t0;
	return vtable_lookup_32bit(tbl16, idx);
#elif (__riscv_v_fixed_vlen * 2) >= (64 * 8)
	vint32_t idx_masked = __riscv_vor(idx.m, 0xFFFFFF00, vint::vl());
	size_t vl = __riscv_vsetvlmax_e8m1();
	vuint8_t idx_bytes0 = __riscv_vreinterpret_u8m1(__riscv_vreinterpret_u32m1(idx_masked));
	vuint8_t idx_bytes1 = __riscv_vadd(idx_bytes0, -32, vl);
	vbool8_t lt32 = __riscv_vmsltu(idx_bytes0, 32, vl);
	vuint8_t res = __riscv_vrgather(tbl.t1, idx_bytes1, vl);
	res = __riscv_vrgather_mu(lt32, res, tbl.t0, idx_bytes0, vl);
	return vint(__riscv_vreinterpret_i32m1(__riscv_vreinterpret_u32m1(res)));
#else /* minimum VLEN is 128 */
	vint32_t idx_masked = __riscv_vor(idx.m, 0xFFFFFF00, vint::vl());
	size_t vl = __riscv_vsetvlmax_e8m1();
	vuint8_t idx_bytes0 = __riscv_vreinterpret_u8m1(__riscv_vreinterpret_u32m1(idx_masked));
	vuint8_t idx_bytes1 = __riscv_vadd(idx_bytes0, -32, vl);
	vuint8_t idx_bytes2 = __riscv_vadd(idx_bytes0, -64, vl);
	vuint8_t idx_bytes3 = __riscv_vadd(idx_bytes0, -96, vl);
	vbool8_t lt32 = __riscv_vmsltu(idx_bytes0, 32, vl);
	vbool8_t lt64 = __riscv_vmsltu(idx_bytes0, 64, vl);
	vbool8_t lt96 = __riscv_vmsltu(idx_bytes0, 96, vl);
	vuint8_t res = __riscv_vrgather(tbl.t1, idx_bytes3, vl);
	res = __riscv_vrgather_mu(lt96, res, tbl.t0, idx_bytes2, vl);
	res = __riscv_vrgather_mu(lt64, res, tbl.t0, idx_bytes1, vl);
	res = __riscv_vrgather_mu(lt32, res, tbl.t0, idx_bytes0, vl);
	return vint(__riscv_vreinterpret_i32m1(__riscv_vreinterpret_u32m1(res)));
#endif
}

/**
 * @brief Return a vector of interleaved RGBA data.
 *
 * Input vectors have the value stored in the bottom 8 bits of each lane,
 * with high bits set to zero.
 *
 * Output vector stores a single RGBA texel packed in each lane.
 */
ASTCENC_SIMD_INLINE vint interleave_rgba8(vint r, vint g, vint b, vint a)
{
	return r + lsl<8>(g) + lsl<16>(b) + lsl<24>(a);
}

/**
 * @brief Store a vector, skipping masked lanes.
 *
 * All masked lanes must be at the end of vector, after all non-masked lanes.
 */
ASTCENC_SIMD_INLINE void store_lanes_masked(uint8_t* base, vint data, vmask mask)
{
	__riscv_vse32(mask.m, reinterpret_cast<int32_t*>(base), data.m, vint::vl());
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void print(vint a)
{
	int v[ASTCENC_SIMD_WIDTH];
	__riscv_vse32(v, a.m, vint::vl());
	printf("v%zu_i32:\n ", (size_t)ASTCENC_SIMD_WIDTH);
	for (size_t i = 0; i < ASTCENC_SIMD_WIDTH; i++)
	{
		printf(" %8d", v[i]);
	}
	puts("");
}

/**
 * @brief Debug function to print a vector of ints.
 */
ASTCENC_SIMD_INLINE void printx(vint a)
{
	int v[ASTCENC_SIMD_WIDTH];
	__riscv_vse32(v, a.m, vint::vl());
	printf("v%zu_i32:\n ", (size_t)ASTCENC_SIMD_WIDTH);
	for (size_t i = 0; i < ASTCENC_SIMD_WIDTH; i++)
	{
		printf(" %08x", v[i]);
	}
	puts("");
}

/**
 * @brief Debug function to print a vector of floats.
 */
ASTCENC_SIMD_INLINE void print(vfloat a)
{
	float v[ASTCENC_SIMD_WIDTH];
	__riscv_vse32(v, a.m, vfloat::vl());
	printf("v%zu_f32:\n ", (size_t)ASTCENC_SIMD_WIDTH);
	for (size_t i = 0; i < ASTCENC_SIMD_WIDTH; i++)
	{
		printf(" %0.4f", static_cast<double>(v[i]));
	}
	puts("");
}

/**
 * @brief Debug function to print a vector of masks.
 */
ASTCENC_SIMD_INLINE void print(vmask a)
{
	print(select(vint(0), vint(1), a));
}

#endif // #ifndef ASTC_VECMATHLIB_RVV_N_H_INCLUDED

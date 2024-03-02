/**************************************************************************/
/*  vector4.h                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VECTOR4_H
#define VECTOR4_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"

class String;

struct _NO_DISCARD_ Vector4 {
	static const int AXIS_COUNT = 4;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
		AXIS_W,
	};

// SSE
#if defined(SSE_WITH_SINGLE_PRECISION)
	using simd_t = __m128;
	using simd_t_arg = simd_t;
#elif defined(SSE_WITH_DOUBLE_PRECISION)
#ifdef AVX_ENABLED
	using simd_t = __m256d;
	using simd_t_arg = __m256d;
#else //AVX_ENABLED
	struct simd_t {
		union {
			struct {
				__m128d low, high;
			};
			__m128d components[2];
		};

		simd_t(__m128d p_low, __m128d p_high) :
				low(p_low), high(p_high) {
		}
	};
	using simd_t_arg = const simd_t &;
	using simd_component_t = __m128d;
#endif //AVX_ENABLED
#endif

// NEON
#if defined(NEON_WITH_SINGLE_PRECISION)
	using simd_t = float32x4_t;
	using simd_t_arg = simd_t;
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	struct simd_t {
		union {
			struct {
				float64x2_t low, high;
			};
			float64x2_t components[2];
		};

		simd_t(const float64x2_t &p_low, const float64x2_t &p_high) :
				low(p_low), high(p_high) {
		}
	};
	using simd_t_arg = const simd_t &;
	using simd_component_t = float64x2_t;
#endif

	union {
#ifdef SIMD_ENABLED
		simd_t simd_value;
#endif // SIMD_ENABLED
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t components[4] = { 0, 0, 0, 0 };
	};

	_FORCE_INLINE_ real_t &operator[](const int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return components[p_axis];
	}
	_FORCE_INLINE_ const real_t &operator[](const int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return components[p_axis];
	}

	Vector4::Axis min_axis_index() const;
	Vector4::Axis max_axis_index() const;

	Vector4 min(const Vector4 &p_vector4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
		return _mm_min_ps(simd_value, p_vector4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
		return _mm256_min_pd(simd_value, p_vector4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
		return Vector4(_mm_min_pd(simd_value.low, p_vector4.simd_value.low),
				_mm_min_pd(simd_value.high, p_vector4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
		return vminq_f32(simd_value, p_vector4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
		return Vector4(vminq_f64(simd_value.low, p_vector4.simd_value.low),
				vminq_f64(simd_value.high, p_vector4.simd_value.high));
#else
		return Vector4(MIN(x, p_vector4.x), MIN(y, p_vector4.y), MIN(z, p_vector4.z), MIN(w, p_vector4.w));
#endif
	}

	Vector4 max(const Vector4 &p_vector4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
		return _mm_max_ps(simd_value, p_vector4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
		return _mm256_max_pd(simd_value, p_vector4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
		return Vector4(_mm_max_pd(simd_value.low, p_vector4.simd_value.low),
				_mm_max_pd(simd_value.high, p_vector4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
		return vmaxq_f32(simd_value, p_vector4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
		return Vector4(vmaxq_f64(simd_value.low, p_vector4.simd_value.low),
				vmaxq_f64(simd_value.high, p_vector4.simd_value.high));
#else
		return Vector4(MAX(x, p_vector4.x), MAX(y, p_vector4.y), MAX(z, p_vector4.z), MAX(w, p_vector4.w));
#endif
	}

	_FORCE_INLINE_ real_t length_squared() const;
	bool is_equal_approx(const Vector4 &p_vec4) const;
	bool is_zero_approx() const;
	bool is_finite() const;
	real_t length() const;
	void normalize();
	Vector4 normalized() const;
	bool is_normalized() const;

	real_t distance_to(const Vector4 &p_to) const;
	real_t distance_squared_to(const Vector4 &p_to) const;
	Vector4 direction_to(const Vector4 &p_to) const;

	Vector4 abs() const;
	Vector4 sign() const;
	Vector4 floor() const;
	Vector4 ceil() const;
	Vector4 round() const;
	Vector4 lerp(const Vector4 &p_to, const real_t p_weight) const;
	Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight) const;
	Vector4 cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight, const real_t &p_b_t, const real_t &p_pre_a_t, const real_t &p_post_b_t) const;

	Vector4 posmod(const real_t p_mod) const;
	Vector4 posmodv(const Vector4 &p_modv) const;
	void snap(const Vector4 &p_step);
	Vector4 snapped(const Vector4 &p_step) const;
	Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const;

	Vector4 inverse() const;
	_FORCE_INLINE_ real_t dot(const Vector4 &p_vec4) const;

	_FORCE_INLINE_ void operator+=(const Vector4 &p_vec4);
	_FORCE_INLINE_ void operator-=(const Vector4 &p_vec4);
	_FORCE_INLINE_ void operator*=(const Vector4 &p_vec4);
	_FORCE_INLINE_ void operator/=(const Vector4 &p_vec4);
	_FORCE_INLINE_ void operator*=(const real_t &s);
	_FORCE_INLINE_ void operator/=(const real_t &s);
	_FORCE_INLINE_ Vector4 operator+(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ Vector4 operator-(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ Vector4 operator*(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ Vector4 operator/(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ Vector4 operator-() const;
	_FORCE_INLINE_ Vector4 operator*(const real_t &s) const;
	_FORCE_INLINE_ Vector4 operator/(const real_t &s) const;

	_FORCE_INLINE_ bool operator==(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool operator!=(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool operator>(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool operator<(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool operator>=(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool operator<=(const Vector4 &p_vec4) const;

	operator String() const;

	_FORCE_INLINE_ Vector4() {}

	_FORCE_INLINE_ Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
#if defined(SSE_WITH_SINGLE_PRECISION)
			simd_value(_mm_set_ps(p_x, p_y, p_z, p_w))
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
			simd_value(_mm256_set_pd(p_x, p_y, p_z, p_w))
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
			simd_value(_mm_set_pd(p_x, p_y), _mm_set_pd(p_z, p_w))
#else
			x(p_x),
			y(p_y),
			z(p_z),
			w(p_w)
#endif
	{
	}

	Vector4(const Vector4 &p_vec4) :
#ifdef SIMD_ENABLED
			simd_value(p_vec4.simd_value)
#else
			x(p_vec4.x),
			y(p_vec4.y),
			z(p_vec4.z),
			w(p_vec4.w)
#endif
	{
	}

#ifdef SIMD_ENABLED
	_FORCE_INLINE_ Vector4(simd_t_arg p_simd_value) :
			simd_value(p_simd_value) {
	}

#if defined(REAL_T_IS_DOUBLE) && (!defined(AVX_ENABLED) || defined(NEON_ENABLED))
	_FORCE_INLINE_ Vector4(simd_component_t p_simd_value_low, simd_component_t p_simd_value_high) :
			simd_value(p_simd_value_low, p_simd_value_high) {
	}
#endif
#endif // SIMD_ENABLED

	void operator=(const Vector4 &p_vec4) {
#ifdef SIMD_ENABLED
#if (defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)) || defined(NEON_WITH_DOUBLE_PRECISION)
		simd_value.low = p_vec4.simd_value.low;
		simd_value.high = p_vec4.simd_value.high;
#else
		simd_value = p_vec4.simd_value;
#endif
#else //SIMD_ENABLED
		x = p_vec4.x;
		y = p_vec4.y;
		z = p_vec4.z;
		w = p_vec4.w;
#endif // SIMD_ENABLED
	}
};

real_t Vector4::dot(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(SSE4_1_ENABLED)
	return _mm_cvtss_f32(_mm_dp_ps(simd_value, p_vec4.simd_value, 0xff));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	__m256d mul = _mm256_mul_pd(simd_value, p_vec4.simd_value);

	__m128d vlow = _mm256_castpd256_pd128(mul);
	__m128d vhigh = _mm256_extractf128_pd(mul, 1); // High 128
	vlow = _mm_add_pd(vlow, vhigh); // Reduce down to 128

	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // Reduce to scalar
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vaddvq_f32(vmulq_f32(simd_value, p_vec4.simd_value));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	float64_t low_sum = vaddvq_f64(vmulq_f64(simd_value.low, p_vec4.simd_value.low));
	float64_t high_sum = vaddvq_f64(vmulq_f64(simd_value.high, p_vec4.simd_value.high));

	return low_sum + high_sum;
#else
	return x * p_vec4.x + y * p_vec4.y + z * p_vec4.z + w * p_vec4.w;
#endif
}

real_t Vector4::length_squared() const {
	return dot(*this);
}

void Vector4::operator+=(const Vector4 &p_vec4) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_add_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_add_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_add_pd(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = _mm_add_pd(simd_value.high, p_vec4.simd_value.high);
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vaddq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vaddq_f64(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = vaddq_f64(simd_value.high, p_vec4.simd_value.high);
#else
	x += p_vec4.x;
	y += p_vec4.y;
	z += p_vec4.z;
	w += p_vec4.w;
#endif
}

void Vector4::operator-=(const Vector4 &p_vec4) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_sub_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_sub_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_sub_pd(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = _mm_sub_pd(simd_value.high, p_vec4.simd_value.high);
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vsubq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vsubq_f64(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = vsubq_f64(simd_value.high, p_vec4.simd_value.high);
#else
	x -= p_vec4.x;
	y -= p_vec4.y;
	z -= p_vec4.z;
	w -= p_vec4.w;
#endif
}

void Vector4::operator*=(const Vector4 &p_vec4) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_mul_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_mul_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_mul_pd(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = _mm_mul_pd(simd_value.high, p_vec4.simd_value.high);
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vmulq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vmulq_f64(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = vmulq_f64(simd_value.high, p_vec4.simd_value.high);
#else
	x *= p_vec4.x;
	y *= p_vec4.y;
	z *= p_vec4.z;
	w *= p_vec4.w;
#endif
}

void Vector4::operator/=(const Vector4 &p_vec4) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_div_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_div_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_div_pd(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = _mm_div_pd(simd_value.high, p_vec4.simd_value.high);
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vdivq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vdivq_f64(simd_value.low, p_vec4.simd_value.low);
	simd_value.high = vdivq_f64(simd_value.high, p_vec4.simd_value.high);
#else
	x /= p_vec4.x;
	y /= p_vec4.y;
	z /= p_vec4.z;
	w /= p_vec4.w;
#endif
}

void Vector4::operator*=(const real_t &s) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_mul_ps(simd_value, _mm_set1_ps(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_mul_pd(simd_value, _mm256_set1_pd(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_mul_pd(simd_value.low, _mm_set1_pd(s));
	simd_value.high = _mm_mul_pd(simd_value.high, _mm_set1_pd(s));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vmulq_n_f32(simd_value, s);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vmulq_n_f64(simd_value.low, s);
	simd_value.high = vmulq_n_f64(simd_value.high, s);
#else
	x *= s;
	y *= s;
	z *= s;
	w *= s;
#endif
}

void Vector4::operator/=(const real_t &s) {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_value = _mm_div_ps(simd_value, _mm_set1_ps(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_value = _mm256_div_pd(simd_value, _mm256_set1_pd(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	simd_value.low = _mm_div_pd(simd_value.low, _mm_set1_pd(s));
	simd_value.high = _mm_div_pd(simd_value.high, _mm_set1_pd(s));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_value = vdivq_f32(simd_value, vdupq_n_f32(s));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_value.low = vdivq_f64(simd_value.low, vdupq_n_f64(s));
	simd_value.high = vdivq_f64(simd_value.high, vdupq_n_f64(s));
#else
	*this *= 1.0f / s;
#endif
}

Vector4 Vector4::operator+(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_add_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_add_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_add_pd(simd_value.low, p_vec4.simd_value.low),
			_mm_add_pd(simd_value.high, p_vec4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vaddq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vaddq_f64(simd_value.low, p_vec4.simd_value.low), vaddq_f64(simd_value.high, p_vec4.simd_value.high));
#else
	return Vector4(x + p_vec4.x, y + p_vec4.y, z + p_vec4.z, w + p_vec4.w);
#endif
}

Vector4 Vector4::operator-(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_sub_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_sub_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_sub_pd(simd_value.low, p_vec4.simd_value.low),
			_mm_sub_pd(simd_value.high, p_vec4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vsubq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vsubq_f64(simd_value.low, p_vec4.simd_value.low),
			vsubq_f64(simd_value.high, p_vec4.simd_value.high));
#else
	return Vector4(x - p_vec4.x, y - p_vec4.y, z - p_vec4.z, w - p_vec4.w);
#endif
}

Vector4 Vector4::operator*(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_mul_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_mul_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_mul_pd(simd_value.low, p_vec4.simd_value.low),
			_mm_mul_pd(simd_value.high, p_vec4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vmulq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vmulq_f64(simd_value.low, p_vec4.simd_value.low),
			vmulq_f64(simd_value.high, p_vec4.simd_value.high));
#else
	return Vector4(x * p_vec4.x, y * p_vec4.y, z * p_vec4.z, w * p_vec4.w);
#endif
}

Vector4 Vector4::operator/(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_div_ps(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_div_pd(simd_value, p_vec4.simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_div_pd(simd_value.low, p_vec4.simd_value.low),
			_mm_div_pd(simd_value.high, p_vec4.simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vdivq_f32(simd_value, p_vec4.simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vdivq_f64(simd_value.low, p_vec4.simd_value.low),
			vdivq_f64(simd_value.high, p_vec4.simd_value.high));
#else
	return Vector4(x / p_vec4.x, y / p_vec4.y, z / p_vec4.z, w / p_vec4.w);
#endif
}

Vector4 Vector4::operator-() const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_sub_ps(_mm_setzero_ps(), simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_sub_pd(_mm256_setzero_pd(), simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_sub_pd(_mm_setzero_pd(), simd_value.low),
			_mm_sub_pd(_mm_setzero_pd(), simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vnegq_f32(simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vnegq_f64(simd_value.low),
			vnegq_f64(simd_value.high));
#else
	return Vector4(-x, -y, -z, -w);
#endif
}

Vector4 Vector4::operator*(const real_t &s) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_mul_ps(simd_value, _mm_set1_ps(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_mul_pd(simd_value, _mm256_set1_pd(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4{ _mm_mul_pd(simd_value.low, _mm_set1_pd(s)),
		_mm_mul_pd(simd_value.high, _mm_set1_pd(s)) };
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vmulq_n_f32(simd_value, s);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vmulq_n_f64(simd_value.low, s), vmulq_n_f64(simd_value.high, s));
#else
	return Vector4(x * s, y * s, z * s, w * s);
#endif
}

Vector4 Vector4::operator/(const real_t &s) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_div_ps(simd_value, _mm_set1_ps(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_div_pd(simd_value, _mm256_set1_pd(s));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4{ _mm_div_pd(simd_value.low, _mm_set1_pd(s)),
		_mm_div_pd(simd_value.high, _mm_set1_pd(s)) };
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vdivq_f32(simd_value, vdupq_n_f32(s));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vdivq_f64(simd_value.low, vdupq_n_f64(s)),
			vdivq_f64(simd_value.high, vdupq_n_f64(s)));
#else
	return *this * (1.0f / s);
#endif
}

bool Vector4::operator==(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION) && !defined(AVX_ENABLED)
	return _mm_movemask_ps(_mm_cmpeq_ps(simd_value, p_vec4.simd_value)) == 0b1111;
#elif defined(SSE_WITH_SINGLE_PRECISION) && defined(AVX_ENABLED)
	return _mm_movemask_ps(_mm_cmp_ps(simd_value, p_vec4.simd_value, _CMP_EQ_OQ)) == 0b1111;
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_movemask_pd(_mm256_cmp_pd(simd_value, p_vec4.simd_value, _CMP_EQ_OQ)) == 0b1111;
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return _mm_movemask_pd(_mm_cmpeq_pd(simd_value.low, p_vec4.simd_value.low)) == 0b11 &&
			_mm_movemask_pd(_mm_cmpeq_pd(simd_value.high, p_vec4.simd_value.high)) == 0b11;
#elif defined(NEON_WITH_SINGLE_PRECISION)
	uint32x4_t cmp = vceqq_f32(simd_value, p_vec4.simd_value);
	for (uint8_t i = 0; i < 4; ++i) {
		if (vgetq_lane_u64(cmp, i) == 0) {
			return false;
		}
	}
	return true;
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	uint64x2_t cmp_low = vceqq_f64(simd_value.low, p_vec4.simd_value.low);
	uint64x2_t cmp_high = vceqq_f64(simd_value.high, p_vec4.simd_value.high);
	for (uint8_t i = 0; i < 2; ++i) {
		if (vgetq_lane_u64(cmp_low, i) == 0) {
			return false;
		}
	}
	for (uint8_t i = 0; i < 2; ++i) {
		if (vgetq_lane_u64(cmp_high, i) == 0) {
			return false;
		}
	}
	return true;
#else
	return x == p_vec4.x && y == p_vec4.y && z == p_vec4.z && w == p_vec4.w;
#endif
}

bool Vector4::operator!=(const Vector4 &p_vec4) const {
#if defined(SSE_WITH_SINGLE_PRECISION) && !defined(AVX_ENABLED)
	return _mm_movemask_ps(_mm_cmpeq_ps(simd_value, p_vec4.simd_value)) != 0b1111;
#elif defined(SSE_WITH_SINGLE_PRECISION) && defined(AVX_ENABLED)
	return _mm_movemask_ps(_mm_cmp_ps(simd_value, p_vec4.simd_value, _CMP_EQ_OQ)) != 0b1111;
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_movemask_pd(_mm256_cmp_pd(simd_value, p_vec4.simd_value, _CMP_EQ_OQ)) != 0b1111;
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return _mm_movemask_pd(_mm_cmpeq_pd(simd_value.low, p_vec4.simd_value.low)) != 0b11 ||
			_mm_movemask_pd(_mm_cmpeq_pd(simd_value.high, p_vec4.simd_value.high)) != 0b11;
#elif defined(NEON_WITH_SINGLE_PRECISION)
	uint32x4_t cmp = vceqq_f32(simd_value, p_vec4.simd_value);
	for (uint8_t i = 0; i < 4; ++i) {
		if (vgetq_lane_u64(cmp, i) == 0) {
			return true;
		}
	}
	return false;
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_component_t cmp_low = vceqq_f64(simd_value.low, p_vec4.simd_value.low);
	simd_component_t cmp_high = vceqq_f64(simd_value.high, p_vec4.simd_value.high);
	for (uint8_t i = 0; i < 2; ++i) {
		if (vgetq_lane_u64(cmp_low, i) == 0) {
			return true;
		}
	}
	for (uint8_t i = 0; i < 2; ++i) {
		if (vgetq_lane_u64(cmp_high, i) == 0) {
			return true;
		}
	}
	return false;
#else
	return x != p_vec4.x || y != p_vec4.y || z != p_vec4.z || w != p_vec4.w;
#endif
}

bool Vector4::operator<(const Vector4 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			if (z == p_v.z) {
				return w < p_v.w;
			}
			return z < p_v.z;
		}
		return y < p_v.y;
	}
	return x < p_v.x;
}

bool Vector4::operator>(const Vector4 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			if (z == p_v.z) {
				return w > p_v.w;
			}
			return z > p_v.z;
		}
		return y > p_v.y;
	}
	return x > p_v.x;
}

bool Vector4::operator<=(const Vector4 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			if (z == p_v.z) {
				return w <= p_v.w;
			}
			return z < p_v.z;
		}
		return y < p_v.y;
	}
	return x < p_v.x;
}

bool Vector4::operator>=(const Vector4 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			if (z == p_v.z) {
				return w >= p_v.w;
			}
			return z > p_v.z;
		}
		return y > p_v.y;
	}
	return x > p_v.x;
}

_FORCE_INLINE_ Vector4 operator*(const float p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(const double p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(const int32_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector4 operator*(const int64_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

#endif // VECTOR4_H

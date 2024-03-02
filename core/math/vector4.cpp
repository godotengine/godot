/**************************************************************************/
/*  vector4.cpp                                                           */
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

#include "vector4.h"

#include "core/string/ustring.h"

Vector4::Axis Vector4::min_axis_index() const {
	uint32_t min_index = 0;
	real_t min_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) <= min_value) {
			min_index = i;
			min_value = operator[](i);
		}
	}
	return Vector4::Axis(min_index);
}

Vector4::Axis Vector4::max_axis_index() const {
	uint32_t max_index = 0;
	real_t max_value = x;
	for (uint32_t i = 1; i < 4; i++) {
		if (operator[](i) > max_value) {
			max_index = i;
			max_value = operator[](i);
		}
	}
	return Vector4::Axis(max_index);
}

bool Vector4::is_equal_approx(const Vector4 &p_vec4) const {
	return Math::is_equal_approx(x, p_vec4.x) && Math::is_equal_approx(y, p_vec4.y) && Math::is_equal_approx(z, p_vec4.z) && Math::is_equal_approx(w, p_vec4.w);
}

bool Vector4::is_zero_approx() const {
	return Math::is_zero_approx(x) && Math::is_zero_approx(y) && Math::is_zero_approx(z) && Math::is_zero_approx(w);
}

bool Vector4::is_finite() const {
	return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
}

real_t Vector4::length() const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(SSE4_1_ENABLED)
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(simd_value, simd_value, 0xff)));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	__m256d mul = _mm256_mul_pd(simd_value, simd_value);

	__m128d vlow = _mm256_castpd256_pd128(mul);
	__m128d vhigh = _mm256_extractf128_pd(mul, 1); // High 128
	vlow = _mm_add_pd(vlow, vhigh); // Reduce down to 128

	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_sd(vlow, high64))); // Reduce to scalar
#elif defined(NEON_WITH_SINGLE_PRECISION)
	float32x4_t mul = vmulq_f32(simd_value, simd_value);
	float32x2_t sum = vdup_n_f32(vaddvq_f32(mul));
	return vget_lane_f32(vsqrt_f32(sum), 0);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	float64x1_t low_sum = vdup_n_f64(vaddvq_f64(vmulq_f64(simd_value.low, simd_value.low)));
	float64x1_t high_sum = vdup_n_f64(vaddvq_f64(vmulq_f64(simd_value.high, simd_value.high)));
	float64x1_t sum = vadd_f64(low_sum, high_sum);
	return vget_lane_f64(vsqrt_f64(sum), 0);
#else
	return Math::sqrt(length_squared());
#endif
}

void Vector4::normalize() {
#ifdef SIMD_ENABLED
	if (*this == Vector4()) {
#if defined(SSE_WITH_SINGLE_PRECISION)
		simd_value = _mm_setzero_ps();
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
		simd_value = _mm256_setzero_pd();
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
		simd_value.low = _mm_setzero_pd();
		simd_value.high = _mm_setzero_pd();
#elif defined(NEON_WITH_SINGLE_PRECISION)
		simd_value = vdupq_n_f32(0.0);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
		simd_value.low = vdupq_n_f64(0.0);
		simd_value.high = vdupq_n_f64(0.0);
#endif
	} else {
#if defined(SSE_WITH_SINGLE_PRECISION)
		simd_value = _mm_div_ps(simd_value, _mm_set1_ps(length()));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
		simd_value = _mm256_div_pd(simd_value, _mm256_set1_pd(length()));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
		simd_component_t len = _mm_set1_pd(length());
		simd_value.low = _mm_div_pd(simd_value.low, len);
		simd_value.high = _mm_div_pd(simd_value.high, len);
#elif defined(NEON_WITH_SINGLE_PRECISION)
		simd_value = vdivq_f32(simd_value, vdupq_n_f32(length()));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
		simd_component_t len = vdupq_n_f64(length());
		simd_value.low = vdivq_f64(simd_value.low, len);
		simd_value.high = vdivq_f64(simd_value.low, len);
#endif
	}

#else // SIMD_ENABLED
	real_t lengthsq = length_squared();
	if (lengthsq == 0) {
		x = y = z = w = 0;
	} else {
		real_t length = Math::sqrt(lengthsq);
		x /= length;
		y /= length;
		z /= length;
		w /= length;
	}
#endif // SIMD_ENABLED
}

Vector4 Vector4::normalized() const {
	Vector4 v = *this;
	v.normalize();
	return v;
}

bool Vector4::is_normalized() const {
	return Math::is_equal_approx(length_squared(), (real_t)1, (real_t)UNIT_EPSILON);
}

real_t Vector4::distance_to(const Vector4 &p_to) const {
	return (p_to - *this).length();
}

real_t Vector4::distance_squared_to(const Vector4 &p_to) const {
	return (p_to - *this).length_squared();
}

Vector4 Vector4::direction_to(const Vector4 &p_to) const {
	Vector4 ret = p_to - *this;
	ret.normalize();
	return ret;
}

Vector4 Vector4::abs() const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), simd_value), simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_max_pd(_mm256_sub_pd(_mm256_setzero_pd(), simd_value), simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4(_mm_max_pd(_mm_sub_pd(_mm_setzero_pd(), simd_value.low), simd_value.low),
			_mm_max_pd(_mm_sub_pd(_mm_setzero_pd(), simd_value.high), simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vabsq_f32(simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vabsq_f64(simd_value.low), vabsq_f64(simd_value.high));
#else
	return Vector4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
#endif
}

Vector4 Vector4::sign() const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	simd_t minus_one = _mm_set1_ps(-1.0f);
	simd_t one = _mm_set1_ps(1.0f);
	return _mm_or_ps(_mm_and_ps(simd_value, minus_one), one);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_t minus_one = _mm256_set1_pd(-1.0);
	simd_t one = _mm256_set1_pd(1.0f);
	return _mm256_or_pd(_mm256_and_pd(simd_value, minus_one), one);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	__m128d minus_one = _mm_set1_pd(-1.0);
	__m128d one = _mm_set1_pd(1.0);
	return Vector4(_mm_or_pd(_mm_and_pd(simd_value.low, minus_one), one),
			_mm_or_pd(_mm_and_pd(simd_value.high, minus_one), one));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_t minus_one = vdupq_n_f32(-1.0f);
	simd_t one = vdupq_n_f32(1.0f);
	return vorrq_s32(vandq_s32(simd_value, minus_one), one);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	simd_component_t minus_one = vdupq_n_f64(-1.0f);
	simd_component_t one = vdupq_n_f64(1.0f);
	return Vector4(vorrq_s64(vandq_s64(simd_value.low, minus_one), one),
			vorrq_s64(vandq_s64(simd_value.high, minus_one), one));
#else
	return Vector4(SIGN(x), SIGN(y), SIGN(z), SIGN(w));
#endif
}

Vector4 Vector4::floor() const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(SSE4_1_ENABLED)
	return _mm_floor_ps(simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_floor_pd(simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	return Vector4(_mm_floor_pd(simd_value.low), _mm_floor_pd(simd_value.high));
#else
	return Vector4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
#endif
}

Vector4 Vector4::ceil() const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(SSE4_1_ENABLED)
	return _mm_ceil_ps(simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_ceil_pd(simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	return Vector4(_mm_ceil_pd(simd_value.low), _mm_ceil_pd(simd_value.high));
#else
	return Vector4(Math::ceil(x), Math::ceil(y), Math::ceil(z), Math::ceil(w));
#endif
}

Vector4 Vector4::round() const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(SSE4_1_ENABLED)
	return _mm_round_ps(simd_value, _MM_ROUND_NEAREST);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_round_pd(simd_value, _MM_ROUND_NEAREST);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	return Vector4(_mm_round_pd(simd_value.low, _MM_ROUND_NEAREST), _mm_round_pd(simd_value.high, _MM_ROUND_NEAREST));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vrndnq_f32(simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vrndnq_f64(simd_value.low), vrndnq_f64(simd_value.high));
#else
	return Vector4(Math::round(x), Math::round(y), Math::round(z), Math::round(w));
#endif
}

Vector4 Vector4::lerp(const Vector4 &p_to, const real_t p_weight) const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	return _mm_add_ps(simd_value, _mm_mul_ps(_mm_sub_ps(p_to.simd_value, simd_value), _mm_set1_ps(p_weight)));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	return _mm256_add_pd(simd_value, _mm256_mul_pd(_mm256_sub_pd(p_to.simd_value, simd_value), _mm256_set1_pd(p_weight)));
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	return Vector4{ _mm_add_pd(simd_value.low, _mm_mul_pd(_mm_sub_pd(p_to.simd_value.low, simd_value.low), _mm_set1_pd(p_weight))),
		_mm_add_pd(simd_value.high, _mm_mul_pd(_mm_sub_pd(p_to.simd_value.high, simd_value.high), _mm_set1_pd(p_weight))) };
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vaddq_f32(simd_value, vmulq_f32(vsubq_f32(p_to.simd_value, simd_value), vdupq_n_f32(p_weight)));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vaddq_f64(simd_value.low, vmulq_f64(vsubq_f64(p_to.simd_value.low, simd_value.low), vdupq_n_f64(p_weight))),
			vaddq_f64(simd_value.high, vmulq_f64(vsubq_f64(p_to.simd_value.high, simd_value.high), vdupq_n_f64(p_weight))));
#else
	Vector4 res = *this;
	res.x = Math::lerp(res.x, p_to.x, p_weight);
	res.y = Math::lerp(res.y, p_to.y, p_weight);
	res.z = Math::lerp(res.z, p_to.z, p_weight);
	res.w = Math::lerp(res.w, p_to.w, p_weight);
	return res;
#endif
}

Vector4 Vector4::cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight) const {
	// TODO SIMD
	Vector4 res = *this;
	res.x = Math::cubic_interpolate(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight);
	res.y = Math::cubic_interpolate(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight);
	res.z = Math::cubic_interpolate(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight);
	res.w = Math::cubic_interpolate(res.w, p_b.w, p_pre_a.w, p_post_b.w, p_weight);
	return res;
}

Vector4 Vector4::cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight, const real_t &p_b_t, const real_t &p_pre_a_t, const real_t &p_post_b_t) const {
	// TODO SIMD
	Vector4 res = *this;
	res.x = Math::cubic_interpolate_in_time(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.y = Math::cubic_interpolate_in_time(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.z = Math::cubic_interpolate_in_time(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.w = Math::cubic_interpolate_in_time(res.w, p_b.w, p_pre_a.w, p_post_b.w, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	return res;
}

Vector4 Vector4::posmod(const real_t p_mod) const {
	// TODO SIMD
	return Vector4(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod), Math::fposmod(w, p_mod));
}

Vector4 Vector4::posmodv(const Vector4 &p_modv) const {
	// TODO SIMD
	return Vector4(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z), Math::fposmod(w, p_modv.w));
}

void Vector4::snap(const Vector4 &p_step) {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(AVX_ENABLED)
	simd_t eq_zero_mask = _mm_cmp_ps(p_step.simd_value, _mm_setzero_ps(), _CMP_EQ_OQ);
	simd_t snapped = _mm_mul_ps(_mm_floor_ps(_mm_add_ps(_mm_div_ps(simd_value, p_step.simd_value), _mm_set1_ps(0.5))), p_step.simd_value);
	simd_value = _mm_blendv_ps(snapped, simd_value, eq_zero_mask);
#elif defined(SSE_WITH_SINGLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	simd_t eq_zero_mask = _mm_cmpeq_ps(p_step.simd_value, _mm_setzero_ps());
	simd_t snapped = _mm_mul_ps(_mm_floor_ps(_mm_add_ps(_mm_div_ps(simd_value, p_step.simd_value), _mm_set1_ps(0.5))), p_step.simd_value);
	simd_value = _mm_blendv_ps(snapped, simd_value, eq_zero_mask);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_t eq_zero_mask = _mm256_cmp_pd(p_step.simd_value, _mm256_setzero_pd(), _CMP_EQ_OQ);
	simd_t snapped = _mm256_mul_pd(_mm256_floor_pd(_mm256_add_pd(_mm256_div_pd(simd_value, p_step.simd_value), _mm256_set1_pd(0.5))), p_step.simd_value);
	simd_value = _mm256_blendv_pd(snapped, simd_value, eq_zero_mask);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	simd_component_t eq_zero_mask_low = _mm_cmpeq_ps(p_step.simd_value.low, _mm_setzero_pd());
	simd_component_t snapped_low = _mm_mul_pd(_mm_floor_pd(_mm_add_pd(_mm_div_pd(simd_value.low, p_step.simd_value.low), _mm_set1_pd(0.5))), p_step.simd_value.low);
	simd_value.low = _mm_blendv_pd(snapped_low, simd_value.low, eq_zero_mask_low);

	simd_component_t eq_zero_mask_high = _mm_cmpeq_ps(p_step.simd_value.high, _mm_setzero_pd());
	simd_component_t snapped_high = _mm_mul_pd(_mm_floor_pd(_mm_add_pd(_mm_div_pd(simd_value.high, p_step.simd_value.high), _mm_set1_pd(0.5))), p_step.simd_value.high);
	simd_value.high = _mm_blendv_pd(snapped_high, simd_value.high, eq_zero_mask_high);
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_t eq_zero_mask = vceqq_f32(p_step.simd_value, vdupq_n_f32(0.0f));
	Vector4 tmp = vaddq_f32(vdivq_f32(simd_value, p_step.simd_value), vdupq_n_f32(0.5));
	simd_t snapped = vmulq_f32(tmp.floor().simd_value, p_step.simd_value);

	for (int i = 0; i < 4; ++i) {
		if ((eq_zero_mask.n128_u32[i] & 0xffff) != 0) {
			simd_value.n128_f32[i] = snapped.n128_f32[i];
		}
	}
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	Vector4 tmp(vaddq_f64(vdivq_f64(simd_value.low, p_step.simd_value.low), vdupq_n_f64(0.5)), vaddq_f64(vdivq_f64(simd_value.high, p_step.simd_value.high), vdupq_n_f64(0.5)));
	Vector4 snapped = tmp.floor() * p_step.simd_value;

	for (int i = 0; i < 2; ++i) {
		simd_component_t eq_zero_mask = vceqq_f64(p_step.simd_value.components[i], vdupq_n_f64(0.0));
		for (int j = 0; j < 2; ++i) {
			if (((eq_zero_mask.n128_u64[j] & 0xffffffff)) != 0) {
				simd_value.components[i].n128_f64[j] = snapped.simd_value.components[i].n128_f32[j];
			}
		}
	}

#else
	x = Math::snapped(x, p_step.x);
	y = Math::snapped(y, p_step.y);
	z = Math::snapped(z, p_step.z);
	w = Math::snapped(w, p_step.w);
#endif
}

Vector4 Vector4::snapped(const Vector4 &p_step) const {
	Vector4 v = *this;
	v.snap(p_step);
	return v;
}

Vector4 Vector4::inverse() const {
#if defined(SSE_WITH_SINGLE_PRECISION)
	constexpr const simd_t identity = simd_t{ 1.0f, 1.0f, 1.0f, 1.0f };
	return _mm_div_ps(identity, simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	constexpr const simd_t identity = simd_t{ 1.0, 1.0, 1.0, 1.0 };
	return _mm256_div_pd(identity, simd_value);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED)
	constexpr const simd_component_t identity = simd_component_t{ 1.0, 1.0 };
	return Vector4(_mm_div_pd(identity, simd_value.low), _mm_div_pd(identity, simd_value.high));
#elif defined(NEON_WITH_SINGLE_PRECISION)
	return vdivq_f32(vdupq_n_f32(1.0f), simd_value);
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	return Vector4(vdivq_f64(vdupq_n_f64(1.0), simd_value.low),
			vdivq_f64(vdupq_n_f64(1.0), simd_value.high));
#else
	return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
#endif
}

Vector4 Vector4::clamp(const Vector4 &p_min, const Vector4 &p_max) const {
#if defined(SSE_WITH_SINGLE_PRECISION) && defined(AVX_ENABLED)
	simd_t less_than_mask = _mm_cmp_ps(p_min.simd_value, p_min.simd_value, _CMP_LT_OQ);
	simd_t greater_than_mask = _mm_cmp_ps(p_min.simd_value, p_min.simd_value, _CMP_GT_OQ);
	return _mm_blendv_ps(_mm_blendv_ps(simd_value, p_min.simd_value, less_than_mask), p_max.simd_value, greater_than_mask);
#elif defined(SSE_WITH_SINGLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	simd_t less_than_mask = _mm_cmplt_ps(p_min.simd_value, p_min.simd_value);
	simd_t greater_than_mask = _mm_cmpgt_ps(p_min.simd_value, p_min.simd_value);
	return _mm_blendv_ps(_mm_blendv_ps(simd_value, p_min.simd_value, less_than_mask), p_max.simd_value, greater_than_mask);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && defined(AVX_ENABLED)
	simd_t less_than_mask = _mm256_cmp_pd(p_min.simd_value, p_min.simd_value, _CMP_LT_OQ);
	simd_t greater_than_mask = _mm256_cmp_pd(p_min.simd_value, p_min.simd_value, _CMP_GT_OQ);
	return _mm256_blendv_pd(_mm256_blendv_pd(simd_value, p_min.simd_value, less_than_mask), p_max.simd_value, greater_than_mask);
#elif defined(SSE_WITH_DOUBLE_PRECISION) && !defined(AVX_ENABLED) && defined(SSE4_1_ENABLED)
	simd_component_t less_than_mask_low = _mm_cmplt_ps(simd_value.low, p_min.simd_value.low);
	simd_component_t greater_than_mask_low = _mm_cmpgt_ps(simd_value.low, p_min.simd_value.low);
	simd_component_t less_than_mask_high = _mm_cmplt_ps(simd_value.high, p_min.simd_value.high);
	simd_component_t greater_than_mask_high = _mm_cmpgt_ps(simd_value.high, p_min.simd_value.high);
	return Vector4{
		_mm_blendv_pd(_mm_blendv_pd(simd_value.low, p_min.simd_value.low, less_than_mask_low), p_max.simd_value.low, greater_than_mask_low),
		_mm_blendv_pd(_mm_blendv_pd(simd_value.high, p_min.simd_value.high, less_than_mask_high), p_max.simd_value.high, greater_than_mask_high)
	};
#elif defined(NEON_WITH_SINGLE_PRECISION)
	simd_t less_than_mask = vcltq_f32(p_min.simd_value, p_min.simd_value);
	simd_t greater_than_mask = vcgtq_f32(p_min.simd_value, p_min.simd_value);
	return Vector4(
			less_than_mask.n128_u32[0] ? p_min.simd_value.n128_f32[0] : (greater_than_mask.n128_u32[0] ? p_max.simd_value.n128_f32[0] : simd_value.n128_f32[0]),
			less_than_mask.n128_u32[1] ? p_min.simd_value.n128_f32[1] : (greater_than_mask.n128_u32[1] ? p_max.simd_value.n128_f32[1] : simd_value.n128_f32[1]),
			less_than_mask.n128_u32[2] ? p_min.simd_value.n128_f32[2] : (greater_than_mask.n128_u32[2] ? p_max.simd_value.n128_f32[2] : simd_value.n128_f32[2]),
			less_than_mask.n128_u32[3] ? p_min.simd_value.n128_f32[3] : (greater_than_mask.n128_u32[3] ? p_max.simd_value.n128_f32[3] : simd_value.n128_f32[3]));
#elif defined(NEON_WITH_DOUBLE_PRECISION)
	uint64x2_t less_than_mask_low = vcltq_f64(p_min.simd_value.low, p_min.simd_value.low);
	uint64x2_t greater_than_mask_low = vcgtq_f32(p_min.simd_value.low, p_min.simd_value.low);
	uint64x2_t less_than_mask_high = vcltq_f64(p_min.simd_value.high, p_min.simd_value.high);
	uint64x2_t greater_than_mask_high = vcgtq_f32(p_min.simd_value.high, p_min.simd_value.high);
	return Vector4(
			less_than_mask_low.n128_u64[0] ? p_min.simd_value.low.n128_f64[0] : (greater_than_mask_low.n128_u32[0] ? p_max.simd_value.low.n128_f64[0] : simd_value.low.n128_f64[0]),
			less_than_mask_low.n128_u64[1] ? p_min.simd_value.low.n128_f64[1] : (greater_than_mask_low.n128_u32[1] ? p_max.simd_value.low.n128_f64[1] : simd_value.low.n128_f64[1]),
			less_than_mask_high.n128_u64[0] ? p_min.simd_value.high.n128_f64[0] : (greater_than_mask_high.n128_u32[0] ? p_max.simd_value.high.n128_f64[0] : simd_value.high.n128_f64[0]),
			less_than_mask_high.n128_u64[1] ? p_min.simd_value.high.n128_f64[1] : (greater_than_mask_high.n128_u32[1] ? p_max.simd_value.high.n128_f64[1] : simd_value.high.n128_f64[1]));
#else
	return Vector4(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z),
			CLAMP(w, p_min.w, p_max.w));
#endif
}

Vector4::operator String() const {
	return "(" + String::num_real(x, false) + ", " + String::num_real(y, false) + ", " + String::num_real(z, false) + ", " + String::num_real(w, false) + ")";
}

static_assert(sizeof(Vector4) == 4 * sizeof(real_t));

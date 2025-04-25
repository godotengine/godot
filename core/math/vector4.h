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

#pragma once

#include "core/math/math_funcs.h"

struct Vector4i;

struct [[nodiscard]] Vector4 {
	static const int AXIS_COUNT = 4;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
		AXIS_W,
	};

	union {
		// NOLINTBEGIN(modernize-use-default-member-init)
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t coord[4] = { 0, 0, 0, 0 };
		// NOLINTEND(modernize-use-default-member-init)
	};

	_FORCE_INLINE_ real_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return coord[p_axis];
	}
	_FORCE_INLINE_ const real_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return coord[p_axis];
	}

	_FORCE_INLINE_ Vector4::Axis min_axis_index() const;
	_FORCE_INLINE_ Vector4::Axis max_axis_index() const;

	_FORCE_INLINE_ Vector4 min(const Vector4 &p_vector4) const {
		return Vector4(MIN(x, p_vector4.x), MIN(y, p_vector4.y), MIN(z, p_vector4.z), MIN(w, p_vector4.w));
	}

	_FORCE_INLINE_ Vector4 minf(real_t p_scalar) const {
		return Vector4(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar), MIN(w, p_scalar));
	}

	_FORCE_INLINE_ Vector4 max(const Vector4 &p_vector4) const {
		return Vector4(MAX(x, p_vector4.x), MAX(y, p_vector4.y), MAX(z, p_vector4.z), MAX(w, p_vector4.w));
	}

	_FORCE_INLINE_ Vector4 maxf(real_t p_scalar) const {
		return Vector4(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar), MAX(w, p_scalar));
	}

	_FORCE_INLINE_ real_t length_squared() const;
	_FORCE_INLINE_ bool is_equal_approx(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool is_zero_approx() const;
	_FORCE_INLINE_ bool is_same(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ bool is_finite() const;
	_FORCE_INLINE_ real_t length() const;
	_FORCE_INLINE_ void normalize();
	_FORCE_INLINE_ Vector4 normalized() const;
	_FORCE_INLINE_ bool is_normalized() const;

	_FORCE_INLINE_ real_t distance_to(const Vector4 &p_to) const;
	_FORCE_INLINE_ real_t distance_squared_to(const Vector4 &p_to) const;
	_FORCE_INLINE_ Vector4 direction_to(const Vector4 &p_to) const;

	_FORCE_INLINE_ Vector4 abs() const;
	_FORCE_INLINE_ Vector4 sign() const;
	_FORCE_INLINE_ Vector4 floor() const;
	_FORCE_INLINE_ Vector4 ceil() const;
	_FORCE_INLINE_ Vector4 round() const;
	_FORCE_INLINE_ Vector4 lerp(const Vector4 &p_to, real_t p_weight) const;
	_FORCE_INLINE_ Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const;
	_FORCE_INLINE_ Vector4 cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

	_FORCE_INLINE_ Vector4 posmod(real_t p_mod) const;
	_FORCE_INLINE_ Vector4 posmodv(const Vector4 &p_modv) const;
	_FORCE_INLINE_ void snap(const Vector4 &p_step);
	_FORCE_INLINE_ void snapf(real_t p_step);
	_FORCE_INLINE_ Vector4 snapped(const Vector4 &p_step) const;
	_FORCE_INLINE_ Vector4 snappedf(real_t p_step) const;
	_FORCE_INLINE_ Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const;
	_FORCE_INLINE_ Vector4 clampf(real_t p_min, real_t p_max) const;

	_FORCE_INLINE_ Vector4 inverse() const;
	_FORCE_INLINE_ real_t dot(const Vector4 &p_vec4) const;

	_FORCE_INLINE_ constexpr void operator+=(const Vector4 &p_vec4);
	_FORCE_INLINE_ constexpr void operator-=(const Vector4 &p_vec4);
	_FORCE_INLINE_ constexpr void operator*=(const Vector4 &p_vec4);
	_FORCE_INLINE_ constexpr void operator/=(const Vector4 &p_vec4);
	_FORCE_INLINE_ constexpr void operator*=(real_t p_s);
	_FORCE_INLINE_ constexpr void operator/=(real_t p_s);
	_FORCE_INLINE_ constexpr Vector4 operator+(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr Vector4 operator-(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr Vector4 operator*(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr Vector4 operator/(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr Vector4 operator-() const;
	_FORCE_INLINE_ constexpr Vector4 operator*(real_t p_s) const;
	_FORCE_INLINE_ constexpr Vector4 operator/(real_t p_s) const;

	_FORCE_INLINE_ constexpr bool operator==(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr bool operator!=(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr bool operator>(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr bool operator<(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr bool operator>=(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ constexpr bool operator<=(const Vector4 &p_vec4) const;

	operator String() const;
	operator Vector4i() const;

	_FORCE_INLINE_ constexpr Vector4() :
			x(0), y(0), z(0), w(0) {}
	_FORCE_INLINE_ constexpr Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x), y(p_y), z(p_z), w(p_w) {}
};

real_t Vector4::dot(const Vector4 &p_vec4) const {
	return x * p_vec4.x + y * p_vec4.y + z * p_vec4.z + w * p_vec4.w;
}

real_t Vector4::length_squared() const {
	return dot(*this);
}

constexpr void Vector4::operator+=(const Vector4 &p_vec4) {
	x += p_vec4.x;
	y += p_vec4.y;
	z += p_vec4.z;
	w += p_vec4.w;
}

constexpr void Vector4::operator-=(const Vector4 &p_vec4) {
	x -= p_vec4.x;
	y -= p_vec4.y;
	z -= p_vec4.z;
	w -= p_vec4.w;
}

constexpr void Vector4::operator*=(const Vector4 &p_vec4) {
	x *= p_vec4.x;
	y *= p_vec4.y;
	z *= p_vec4.z;
	w *= p_vec4.w;
}

constexpr void Vector4::operator/=(const Vector4 &p_vec4) {
	x /= p_vec4.x;
	y /= p_vec4.y;
	z /= p_vec4.z;
	w /= p_vec4.w;
}
constexpr void Vector4::operator*=(real_t p_s) {
	x *= p_s;
	y *= p_s;
	z *= p_s;
	w *= p_s;
}

constexpr void Vector4::operator/=(real_t p_s) {
	*this *= (1 / p_s);
}

constexpr Vector4 Vector4::operator+(const Vector4 &p_vec4) const {
	return Vector4(x + p_vec4.x, y + p_vec4.y, z + p_vec4.z, w + p_vec4.w);
}

constexpr Vector4 Vector4::operator-(const Vector4 &p_vec4) const {
	return Vector4(x - p_vec4.x, y - p_vec4.y, z - p_vec4.z, w - p_vec4.w);
}

constexpr Vector4 Vector4::operator*(const Vector4 &p_vec4) const {
	return Vector4(x * p_vec4.x, y * p_vec4.y, z * p_vec4.z, w * p_vec4.w);
}

constexpr Vector4 Vector4::operator/(const Vector4 &p_vec4) const {
	return Vector4(x / p_vec4.x, y / p_vec4.y, z / p_vec4.z, w / p_vec4.w);
}

constexpr Vector4 Vector4::operator-() const {
	return Vector4(-x, -y, -z, -w);
}

constexpr Vector4 Vector4::operator*(real_t p_s) const {
	return Vector4(x * p_s, y * p_s, z * p_s, w * p_s);
}

constexpr Vector4 Vector4::operator/(real_t p_s) const {
	return *this * (1 / p_s);
}

constexpr bool Vector4::operator==(const Vector4 &p_vec4) const {
	return x == p_vec4.x && y == p_vec4.y && z == p_vec4.z && w == p_vec4.w;
}

constexpr bool Vector4::operator!=(const Vector4 &p_vec4) const {
	return x != p_vec4.x || y != p_vec4.y || z != p_vec4.z || w != p_vec4.w;
}

constexpr bool Vector4::operator<(const Vector4 &p_v) const {
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

constexpr bool Vector4::operator>(const Vector4 &p_v) const {
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

constexpr bool Vector4::operator<=(const Vector4 &p_v) const {
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

constexpr bool Vector4::operator>=(const Vector4 &p_v) const {
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

_FORCE_INLINE_ constexpr Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ constexpr Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ constexpr Vector4 operator*(int32_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ constexpr Vector4 operator*(int64_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

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

bool Vector4::is_same(const Vector4 &p_vec4) const {
	return Math::is_same(x, p_vec4.x) && Math::is_same(y, p_vec4.y) && Math::is_same(z, p_vec4.z) && Math::is_same(w, p_vec4.w);
}

bool Vector4::is_zero_approx() const {
	return Math::is_zero_approx(x) && Math::is_zero_approx(y) && Math::is_zero_approx(z) && Math::is_zero_approx(w);
}

bool Vector4::is_finite() const {
	return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
}

real_t Vector4::length() const {
	return Math::sqrt(length_squared());
}

void Vector4::normalize() {
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
	Vector4 ret(p_to.x - x, p_to.y - y, p_to.z - z, p_to.w - w);
	ret.normalize();
	return ret;
}

Vector4 Vector4::abs() const {
	return Vector4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
}

Vector4 Vector4::sign() const {
	return Vector4(SIGN(x), SIGN(y), SIGN(z), SIGN(w));
}

Vector4 Vector4::floor() const {
	return Vector4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
}

Vector4 Vector4::ceil() const {
	return Vector4(Math::ceil(x), Math::ceil(y), Math::ceil(z), Math::ceil(w));
}

Vector4 Vector4::round() const {
	return Vector4(Math::round(x), Math::round(y), Math::round(z), Math::round(w));
}

Vector4 Vector4::lerp(const Vector4 &p_to, real_t p_weight) const {
	Vector4 res = *this;
	res.x = Math::lerp(res.x, p_to.x, p_weight);
	res.y = Math::lerp(res.y, p_to.y, p_weight);
	res.z = Math::lerp(res.z, p_to.z, p_weight);
	res.w = Math::lerp(res.w, p_to.w, p_weight);
	return res;
}

Vector4 Vector4::cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const {
	Vector4 res = *this;
	res.x = Math::cubic_interpolate(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight);
	res.y = Math::cubic_interpolate(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight);
	res.z = Math::cubic_interpolate(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight);
	res.w = Math::cubic_interpolate(res.w, p_b.w, p_pre_a.w, p_post_b.w, p_weight);
	return res;
}

Vector4 Vector4::cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
	Vector4 res = *this;
	res.x = Math::cubic_interpolate_in_time(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.y = Math::cubic_interpolate_in_time(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.z = Math::cubic_interpolate_in_time(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.w = Math::cubic_interpolate_in_time(res.w, p_b.w, p_pre_a.w, p_post_b.w, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	return res;
}

Vector4 Vector4::posmod(real_t p_mod) const {
	return Vector4(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod), Math::fposmod(w, p_mod));
}

Vector4 Vector4::posmodv(const Vector4 &p_modv) const {
	return Vector4(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z), Math::fposmod(w, p_modv.w));
}

void Vector4::snap(const Vector4 &p_step) {
	x = Math::snapped(x, p_step.x);
	y = Math::snapped(y, p_step.y);
	z = Math::snapped(z, p_step.z);
	w = Math::snapped(w, p_step.w);
}

void Vector4::snapf(real_t p_step) {
	x = Math::snapped(x, p_step);
	y = Math::snapped(y, p_step);
	z = Math::snapped(z, p_step);
	w = Math::snapped(w, p_step);
}

Vector4 Vector4::snapped(const Vector4 &p_step) const {
	Vector4 v = *this;
	v.snap(p_step);
	return v;
}

Vector4 Vector4::snappedf(real_t p_step) const {
	Vector4 v = *this;
	v.snapf(p_step);
	return v;
}

Vector4 Vector4::inverse() const {
	return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
}

Vector4 Vector4::clamp(const Vector4 &p_min, const Vector4 &p_max) const {
	return Vector4(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z),
			CLAMP(w, p_min.w, p_max.w));
}

Vector4 Vector4::clampf(real_t p_min, real_t p_max) const {
	return Vector4(
			CLAMP(x, p_min, p_max),
			CLAMP(y, p_min, p_max),
			CLAMP(z, p_min, p_max),
			CLAMP(w, p_min, p_max));
}

template <>
struct is_zero_constructible<Vector4> : std::true_type {};

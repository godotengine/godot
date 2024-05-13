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
#include "core/math/math_defs.h"
#include "core/typedefs.h"

class String;
struct Vector4i;

struct _NO_DISCARD_ Vector4 {
	static constexpr int AXIS_COUNT = 4;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
		AXIS_W,
	};

	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t components[4] = { 0, 0, 0, 0 };
	};

	constexpr const real_t &operator[](size_t p_axis) const;
	constexpr real_t &operator[](size_t p_axis);

	Vector4::Axis min_axis_index() const;
	Vector4::Axis max_axis_index() const;

	Vector4 min(const Vector4 &p_vector4) const {
		return Vector4(MIN(x, p_vector4.x), MIN(y, p_vector4.y), MIN(z, p_vector4.z), MIN(w, p_vector4.w));
	}

	Vector4 minf(real_t p_scalar) const {
		return Vector4(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar), MIN(w, p_scalar));
	}

	Vector4 max(const Vector4 &p_vector4) const {
		return Vector4(MAX(x, p_vector4.x), MAX(y, p_vector4.y), MAX(z, p_vector4.z), MAX(w, p_vector4.w));
	}

	Vector4 maxf(real_t p_scalar) const {
		return Vector4(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar), MAX(w, p_scalar));
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
	Vector4 lerp(const Vector4 &p_to, real_t p_weight) const;
	Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight) const;
	Vector4 cubic_interpolate_in_time(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

	Vector4 posmod(real_t p_mod) const;
	Vector4 posmodv(const Vector4 &p_modv) const;
	void snap(const Vector4 &p_step);
	void snapf(real_t p_step);
	Vector4 snapped(const Vector4 &p_step) const;
	Vector4 snappedf(real_t p_step) const;
	Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const;
	Vector4 clampf(real_t p_min, real_t p_max) const;

	Vector4 inverse() const;
	_FORCE_INLINE_ real_t dot(const Vector4 &p_vec4) const;

	constexpr Vector4 &operator+=(const Vector4 &p_other);
	constexpr Vector4 operator+(const Vector4 &p_other) const;
	constexpr Vector4 &operator-=(const Vector4 &p_other);
	constexpr Vector4 operator-(const Vector4 &p_other) const;
	constexpr Vector4 &operator*=(const Vector4 &p_other);
	constexpr Vector4 operator*(const Vector4 &p_other) const;
	constexpr Vector4 &operator/=(const Vector4 &p_other);
	constexpr Vector4 operator/(const Vector4 &p_other) const;

	constexpr Vector4 &operator*=(real_t p_scalar);
	constexpr Vector4 operator*(real_t p_scalar) const;
	constexpr Vector4 &operator/=(real_t p_scalar);
	constexpr Vector4 operator/(real_t p_scalar) const;

	constexpr Vector4 operator-() const;

	constexpr bool operator==(const Vector4 &p_other) const;
	constexpr bool operator!=(const Vector4 &p_other) const;
	constexpr bool operator>(const Vector4 &p_other) const;
	constexpr bool operator<(const Vector4 &p_other) const;
	constexpr bool operator>=(const Vector4 &p_other) const;
	constexpr bool operator<=(const Vector4 &p_other) const;

	operator String() const;
	operator Vector4i() const;

	constexpr Vector4() :
			x(0), y(0), z(0), w(0) {}

	constexpr Vector4(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x), y(p_y), z(p_z), w(p_w) {}
};

constexpr const real_t &Vector4::operator[](size_t p_axis) const {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_axis, AXIS_COUNT);
	}
#endif
	switch (p_axis) {
		case AXIS_X:
			return x;
		case AXIS_Y:
			return y;
		case AXIS_Z:
			return z;
		case AXIS_W:
			return w;
		default:
			return components[p_axis];
	}
}

constexpr real_t &Vector4::operator[](size_t p_axis) {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_axis, AXIS_COUNT);
	}
#endif
	switch (p_axis) {
		case AXIS_X:
			return x;
		case AXIS_Y:
			return y;
		case AXIS_Z:
			return z;
		case AXIS_W:
			return w;
		default:
			return components[p_axis];
	}
}

constexpr Vector4 &Vector4::operator+=(const Vector4 &p_other) {
	x += p_other.x;
	y += p_other.y;
	z += p_other.z;
	w += p_other.w;
	return *this;
}

constexpr Vector4 Vector4::operator+(const Vector4 &p_other) const {
	return Vector4(x + p_other.x, y + p_other.y, z + p_other.z, w + p_other.w);
}

constexpr Vector4 &Vector4::operator-=(const Vector4 &p_other) {
	x -= p_other.x;
	y -= p_other.y;
	z -= p_other.z;
	w -= p_other.w;
	return *this;
}

constexpr Vector4 Vector4::operator-(const Vector4 &p_other) const {
	return Vector4(x - p_other.x, y - p_other.y, z - p_other.z, w - p_other.w);
}

constexpr Vector4 &Vector4::operator*=(const Vector4 &p_other) {
	x *= p_other.x;
	y *= p_other.y;
	z *= p_other.z;
	w *= p_other.w;
	return *this;
}

constexpr Vector4 Vector4::operator*(const Vector4 &p_other) const {
	return Vector4(x * p_other.x, y * p_other.y, z * p_other.z, w * p_other.w);
}

constexpr Vector4 &Vector4::operator/=(const Vector4 &p_other) {
	x /= p_other.x;
	y /= p_other.y;
	z /= p_other.z;
	w /= p_other.w;
	return *this;
}

constexpr Vector4 Vector4::operator/(const Vector4 &p_other) const {
	return Vector4(x / p_other.x, y / p_other.y, z / p_other.z, w / p_other.w);
}

constexpr Vector4 &Vector4::operator*=(real_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	w *= p_scalar;
	return *this;
}

constexpr Vector4 Vector4::operator*(real_t p_scalar) const {
	return Vector4(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
}

constexpr Vector4 &Vector4::operator/=(real_t p_scalar) {
	*this *= 1.0f / p_scalar;
	return *this;
}

constexpr Vector4 Vector4::operator/(real_t p_scalar) const {
	return *this * (1.0f / p_scalar);
}

constexpr Vector4 Vector4::operator-() const {
	return Vector4(-x, -y, -z, -w);
}

constexpr bool Vector4::operator==(const Vector4 &p_other) const {
	return x == p_other.x && y == p_other.y && z == p_other.z && w == p_other.w;
}

constexpr bool Vector4::operator!=(const Vector4 &p_other) const {
	return x != p_other.x || y != p_other.y || z != p_other.z || w != p_other.w;
}

constexpr bool Vector4::operator<(const Vector4 &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w < p_other.w;
			}
			return z < p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector4::operator>(const Vector4 &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w > p_other.w;
			}
			return z > p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

constexpr bool Vector4::operator<=(const Vector4 &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w <= p_other.w;
			}
			return z < p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector4::operator>=(const Vector4 &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w >= p_other.w;
			}
			return z > p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

// Multiplication operators required to workaround issues with LLVM using implicit conversion
// to Vector4i instead for integers where it should not.

constexpr Vector4 operator*(float p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

constexpr Vector4 operator*(double p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

constexpr Vector4 operator*(int32_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

constexpr Vector4 operator*(int64_t p_scalar, const Vector4 &p_vec) {
	return p_vec * p_scalar;
}

real_t Vector4::dot(const Vector4 &p_vec4) const {
	return x * p_vec4.x + y * p_vec4.y + z * p_vec4.z + w * p_vec4.w;
}

real_t Vector4::length_squared() const {
	return dot(*this);
}

#endif // VECTOR4_H

/*************************************************************************/
/*  vector4.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

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

	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};

		real_t coord[4] = { 0 };
	};

	void set_axis(const int p_axis, const real_t p_value);
	real_t get_axis(const int p_axis) const;

	_FORCE_INLINE_ void set_all(const real_t p_value);

	_FORCE_INLINE_ Vector4::Axis min_axis_index() const;
	_FORCE_INLINE_ Vector4::Axis max_axis_index() const;

	_FORCE_INLINE_ real_t length() const;
	_FORCE_INLINE_ real_t length_squared() const;

	_FORCE_INLINE_ void normalize();
	_FORCE_INLINE_ Vector4 normalized() const;
	_FORCE_INLINE_ bool is_normalized() const;
	_FORCE_INLINE_ Vector4 inverse() const;
	Vector4 limit_length(const real_t p_len = 1.0) const;

	_FORCE_INLINE_ void zero();

	void snap(const Vector4 &p_val);
	Vector4 snapped(const Vector4 &p_val) const;

	/* Static Methods between 2 Vector4s */

	_FORCE_INLINE_ Vector4 lerp(const Vector4 &p_to, const real_t p_weight) const;
	Vector4 cubic_interpolate(const Vector4 &p_b, const Vector4 &p_pre_a, const Vector4 &p_post_b, const real_t p_weight) const;
	Vector4 move_toward(const Vector4 &p_to, const real_t p_delta) const;

	_FORCE_INLINE_ real_t dot(const Vector4 &p_with) const;

	_FORCE_INLINE_ Vector4 abs() const;
	_FORCE_INLINE_ Vector4 floor() const;
	_FORCE_INLINE_ Vector4 sign() const;
	_FORCE_INLINE_ Vector4 ceil() const;
	_FORCE_INLINE_ Vector4 round() const;
	Vector4 clamp(const Vector4 &p_min, const Vector4 &p_max) const;

	_FORCE_INLINE_ real_t distance_to(const Vector4 &p_to) const;
	_FORCE_INLINE_ real_t distance_squared_to(const Vector4 &p_to) const;

	_FORCE_INLINE_ Vector4 posmod(const real_t p_mod) const;
	_FORCE_INLINE_ Vector4 posmodv(const Vector4 &p_modv) const;
	_FORCE_INLINE_ Vector4 project(const Vector4 &p_to) const;

	_FORCE_INLINE_ Vector4 direction_to(const Vector4 &p_to) const;

	_FORCE_INLINE_ Vector4 bounce(const Vector4 &p_normal) const;
	_FORCE_INLINE_ Vector4 reflect(const Vector4 &p_normal) const;

	bool is_equal_approx(const Vector4 &p_v) const;

	/* Operators */

	_FORCE_INLINE_ const real_t &operator[](const int p_axis) const;
	_FORCE_INLINE_ real_t &operator[](const int p_axis);

	_FORCE_INLINE_ Vector4 &operator+=(const Vector4 &p_v);
	_FORCE_INLINE_ Vector4 operator+(const Vector4 &p_v) const;
	_FORCE_INLINE_ Vector4 &operator-=(const Vector4 &p_v);
	_FORCE_INLINE_ Vector4 operator-(const Vector4 &p_v) const;
	_FORCE_INLINE_ Vector4 &operator*=(const Vector4 &p_v);
	_FORCE_INLINE_ Vector4 operator*(const Vector4 &p_v) const;
	_FORCE_INLINE_ Vector4 &operator/=(const Vector4 &p_v);
	_FORCE_INLINE_ Vector4 operator/(const Vector4 &p_v) const;

	_FORCE_INLINE_ Vector4 &operator*=(const real_t p_scalar);
	_FORCE_INLINE_ Vector4 operator*(const real_t p_scalar) const;
	_FORCE_INLINE_ Vector4 &operator/=(const real_t p_scalar);
	_FORCE_INLINE_ Vector4 operator/(const real_t p_scalar) const;

	_FORCE_INLINE_ Vector4 operator-() const;

	_FORCE_INLINE_ bool operator==(const Vector4 &p_v) const;
	_FORCE_INLINE_ bool operator!=(const Vector4 &p_v) const;
	_FORCE_INLINE_ bool operator<(const Vector4 &p_v) const;
	_FORCE_INLINE_ bool operator<=(const Vector4 &p_v) const;
	_FORCE_INLINE_ bool operator>(const Vector4 &p_v) const;
	_FORCE_INLINE_ bool operator>=(const Vector4 &p_v) const;

	operator String() const;

	_FORCE_INLINE_ Vector4() {}
	_FORCE_INLINE_ Vector4(const real_t p_x, const real_t p_y, const real_t p_z, const real_t p_w) {
		x = p_x;
		y = p_y;
		z = p_z;
		w = p_w;
	}
};

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

real_t Vector4::dot(const Vector4 &p_with) const {
	return x * p_with.x + y * p_with.y + z * p_with.z + w * p_with.w;
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

Vector4 Vector4::lerp(const Vector4 &p_to, const real_t p_weight) const {
	return Vector4(
			x + (p_weight * (p_to.x - x)),
			y + (p_weight * (p_to.y - y)),
			z + (p_weight * (p_to.z - z)),
			w + (p_weight * (p_to.w - w)));
}

real_t Vector4::distance_to(const Vector4 &p_to) const {
	return (p_to - *this).length();
}

real_t Vector4::distance_squared_to(const Vector4 &p_to) const {
	return (p_to - *this).length_squared();
}

Vector4 Vector4::posmod(const real_t p_mod) const {
	return Vector4(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod), Math::fposmod(w, p_mod));
}

Vector4 Vector4::posmodv(const Vector4 &p_modv) const {
	return Vector4(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z), Math::fposmod(w, p_modv.w));
}

Vector4 Vector4::project(const Vector4 &p_to) const {
	return p_to * (dot(p_to) / p_to.length_squared());
}

Vector4 Vector4::direction_to(const Vector4 &p_to) const {
	Vector4 ret(p_to.x - x, p_to.y - y, p_to.z - z, p_to.w - w);
	ret.normalize();
	return ret;
}

_FORCE_INLINE_ real_t vec4_dot(const Vector4 &p_a, const Vector4 &p_b) {
	return p_a.dot(p_b);
}

void Vector4::set_all(const real_t p_value) {
	x = y = z = w = p_value;
}

Vector4::Axis Vector4::min_axis_index() const {
	if (z < y) {
		return (z < x) ? ((z < w) ? Vector4::AXIS_Z : Vector4::AXIS_W) : ((w < x) ? Vector4::AXIS_W : Vector4::AXIS_X);
	} else {
		return (y < x) ? ((y < w) ? Vector4::AXIS_Y : Vector4::AXIS_W) : ((w < x) ? Vector4::AXIS_W : Vector4::AXIS_X);
	}
}

Vector4::Axis Vector4::max_axis_index() const {
	if (x > y) {
		return (x > w) ? ((x > z) ? Vector4::AXIS_X : Vector4::AXIS_Z) : ((z > w) ? Vector4::AXIS_Z : Vector4::AXIS_W);
	} else {
		return (y > w) ? ((y > z) ? Vector4::AXIS_Y : Vector4::AXIS_Z) : ((z > w) ? Vector4::AXIS_Z : Vector4::AXIS_W);
	}
}

real_t Vector4::length() const {
	real_t x2 = x * x;
	real_t y2 = y * y;
	real_t z2 = z * z;
	real_t w2 = w * w;

	return Math::sqrt(x2 + y2 + z2 + w2);
}

real_t Vector4::length_squared() const {
	real_t x2 = x * x;
	real_t y2 = y * y;
	real_t z2 = z * z;
	real_t w2 = w * w;

	return x2 + y2 + z2 + w2;
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
	// use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON);
}

Vector4 Vector4::inverse() const {
	return Vector4(1.0f / x, 1.0f / y, 1.0f / z, 1.0f / w);
}

void Vector4::zero() {
	x = y = z = w = 0;
}

Vector4 Vector4::bounce(const Vector4 &p_normal) const {
	return -reflect(p_normal);
}

Vector4 Vector4::reflect(const Vector4 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector4(), "The normal Vector4 must be normalized.");
#endif
	return 2.0f * p_normal * this->dot(p_normal) - *this;
}

/* Operators */

const real_t &Vector4::operator[](const int p_axis) const {
	DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
	return coord[p_axis];
}

real_t &Vector4::operator[](const int p_axis) {
	DEV_ASSERT((unsigned int)p_axis < AXIS_COUNT);
	return coord[p_axis];
}

Vector4 &Vector4::operator+=(const Vector4 &p_v) {
	x += p_v.x;
	y += p_v.y;
	z += p_v.z;
	w += p_v.w;
	return *this;
}

Vector4 Vector4::operator+(const Vector4 &p_v) const {
	return Vector4(x + p_v.x, y + p_v.y, z + p_v.z, w + p_v.w);
}

Vector4 &Vector4::operator-=(const Vector4 &p_v) {
	x -= p_v.x;
	y -= p_v.y;
	z -= p_v.z;
	w -= p_v.w;
	return *this;
}

Vector4 Vector4::operator-(const Vector4 &p_v) const {
	return Vector4(x - p_v.x, y - p_v.y, z - p_v.z, w - p_v.w);
}

Vector4 &Vector4::operator*=(const Vector4 &p_v) {
	x *= p_v.x;
	y *= p_v.y;
	z *= p_v.z;
	w *= p_v.w;
	return *this;
}

Vector4 Vector4::operator*(const Vector4 &p_v) const {
	return Vector4(x * p_v.x, y * p_v.y, z * p_v.z, w * p_v.w);
}

Vector4 &Vector4::operator/=(const Vector4 &p_v) {
	x /= p_v.x;
	y /= p_v.y;
	z /= p_v.z;
	w /= p_v.w;
	return *this;
}

Vector4 Vector4::operator/(const Vector4 &p_v) const {
	return Vector4(x / p_v.x, y / p_v.y, z / p_v.z, w / p_v.w);
}

Vector4 &Vector4::operator*=(const real_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	w *= p_scalar;
	return *this;
}

Vector4 Vector4::operator*(const real_t p_scalar) const {
	return Vector4(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
}

Vector4 &Vector4::operator/=(const real_t p_scalar) {
	x /= p_scalar;
	y /= p_scalar;
	z /= p_scalar;
	w /= p_scalar;
	return *this;
}

Vector4 Vector4::operator/(const real_t p_scalar) const {
	return Vector4(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar);
}

Vector4 Vector4::operator-() const {
	return Vector4(-x, -y, -z, -w);
}

bool Vector4::operator==(const Vector4 &p_v) const {
	return x == p_v.x && y == p_v.y && z == p_v.z && w == p_v.w;
}

bool Vector4::operator!=(const Vector4 &p_v) const {
	return x != p_v.x || y != p_v.y || z != p_v.z || w != p_v.w;
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

#endif // VECTOR4_H

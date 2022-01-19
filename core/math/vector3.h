/*************************************************************************/
/*  vector3.h                                                            */
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

#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/math/math_funcs.h"
#include "core/math/vector2.h"
#include "core/math/vector3i.h"
#include "core/string/ustring.h"
class Basis;

struct _NO_DISCARD_ Vector3 {
	static const int AXIS_COUNT = 3;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
		};

		real_t coord[3] = { 0 };
	};

	_FORCE_INLINE_ const real_t &operator[](const int p_axis) const {
		return coord[p_axis];
	}

	_FORCE_INLINE_ real_t &operator[](const int p_axis) {
		return coord[p_axis];
	}

	void set_axis(const int p_axis, const real_t p_value);
	real_t get_axis(const int p_axis) const;

	_FORCE_INLINE_ void set_all(const real_t p_value) {
		x = y = z = p_value;
	}

	_FORCE_INLINE_ Vector3::Axis min_axis_index() const {
		return x < y ? (x < z ? Vector3::AXIS_X : Vector3::AXIS_Z) : (y < z ? Vector3::AXIS_Y : Vector3::AXIS_Z);
	}

	_FORCE_INLINE_ Vector3::Axis max_axis_index() const {
		return x < y ? (y < z ? Vector3::AXIS_Z : Vector3::AXIS_Y) : (x < z ? Vector3::AXIS_Z : Vector3::AXIS_X);
	}

	_FORCE_INLINE_ real_t length() const;
	_FORCE_INLINE_ real_t length_squared() const;

	_FORCE_INLINE_ void normalize();
	_FORCE_INLINE_ Vector3 normalized() const;
	_FORCE_INLINE_ bool is_normalized() const;
	_FORCE_INLINE_ Vector3 inverse() const;
	Vector3 limit_length(const real_t p_len = 1.0) const;

	_FORCE_INLINE_ void zero();

	void snap(const Vector3 p_val);
	Vector3 snapped(const Vector3 p_val) const;

	void rotate(const Vector3 &p_axis, const real_t p_phi);
	Vector3 rotated(const Vector3 &p_axis, const real_t p_phi) const;

	/* Static Methods between 2 vector3s */

	_FORCE_INLINE_ Vector3 lerp(const Vector3 &p_to, const real_t p_weight) const;
	_FORCE_INLINE_ Vector3 slerp(const Vector3 &p_to, const real_t p_weight) const;
	Vector3 cubic_interpolate(const Vector3 &p_b, const Vector3 &p_pre_a, const Vector3 &p_post_b, const real_t p_weight) const;
	Vector3 move_toward(const Vector3 &p_to, const real_t p_delta) const;

	_FORCE_INLINE_ Vector2 octahedron_encode() const {
		Vector3 n = *this;
		n /= Math::abs(n.x) + Math::abs(n.y) + Math::abs(n.z);
		Vector2 o;
		if (n.z >= 0.0) {
			o.x = n.x;
			o.y = n.y;
		} else {
			o.x = (1.0 - Math::abs(n.y)) * (n.x >= 0.0 ? 1.0 : -1.0);
			o.y = (1.0 - Math::abs(n.x)) * (n.y >= 0.0 ? 1.0 : -1.0);
		}
		o.x = o.x * 0.5 + 0.5;
		o.y = o.y * 0.5 + 0.5;
		return o;
	}

	static _FORCE_INLINE_ Vector3 octahedron_decode(const Vector2 &p_oct) {
		Vector2 f(p_oct.x * 2.0 - 1.0, p_oct.y * 2.0 - 1.0);
		Vector3 n(f.x, f.y, 1.0f - Math::abs(f.x) - Math::abs(f.y));
		float t = CLAMP(-n.z, 0.0, 1.0);
		n.x += n.x >= 0 ? -t : t;
		n.y += n.y >= 0 ? -t : t;
		return n.normalized();
	}

	_FORCE_INLINE_ Vector3 cross(const Vector3 &p_with) const;
	_FORCE_INLINE_ real_t dot(const Vector3 &p_with) const;
	Basis outer(const Vector3 &p_with) const;

	_FORCE_INLINE_ Vector3 abs() const;
	_FORCE_INLINE_ Vector3 floor() const;
	_FORCE_INLINE_ Vector3 sign() const;
	_FORCE_INLINE_ Vector3 ceil() const;
	_FORCE_INLINE_ Vector3 round() const;
	Vector3 clamp(const Vector3 &p_min, const Vector3 &p_max) const;

	_FORCE_INLINE_ real_t distance_to(const Vector3 &p_to) const;
	_FORCE_INLINE_ real_t distance_squared_to(const Vector3 &p_to) const;

	_FORCE_INLINE_ Vector3 posmod(const real_t p_mod) const;
	_FORCE_INLINE_ Vector3 posmodv(const Vector3 &p_modv) const;
	_FORCE_INLINE_ Vector3 project(const Vector3 &p_to) const;

	_FORCE_INLINE_ real_t angle_to(const Vector3 &p_to) const;
	_FORCE_INLINE_ real_t signed_angle_to(const Vector3 &p_to, const Vector3 &p_axis) const;
	_FORCE_INLINE_ Vector3 direction_to(const Vector3 &p_to) const;

	_FORCE_INLINE_ Vector3 slide(const Vector3 &p_normal) const;
	_FORCE_INLINE_ Vector3 bounce(const Vector3 &p_normal) const;
	_FORCE_INLINE_ Vector3 reflect(const Vector3 &p_normal) const;

	bool is_equal_approx(const Vector3 &p_v) const;

	/* Operators */

	_FORCE_INLINE_ Vector3 &operator+=(const Vector3 &p_v);
	_FORCE_INLINE_ Vector3 operator+(const Vector3 &p_v) const;
	_FORCE_INLINE_ Vector3 &operator-=(const Vector3 &p_v);
	_FORCE_INLINE_ Vector3 operator-(const Vector3 &p_v) const;
	_FORCE_INLINE_ Vector3 &operator*=(const Vector3 &p_v);
	_FORCE_INLINE_ Vector3 operator*(const Vector3 &p_v) const;
	_FORCE_INLINE_ Vector3 &operator/=(const Vector3 &p_v);
	_FORCE_INLINE_ Vector3 operator/(const Vector3 &p_v) const;

	_FORCE_INLINE_ Vector3 &operator*=(const real_t p_scalar);
	_FORCE_INLINE_ Vector3 operator*(const real_t p_scalar) const;
	_FORCE_INLINE_ Vector3 &operator/=(const real_t p_scalar);
	_FORCE_INLINE_ Vector3 operator/(const real_t p_scalar) const;

	_FORCE_INLINE_ Vector3 operator-() const;

	_FORCE_INLINE_ bool operator==(const Vector3 &p_v) const;
	_FORCE_INLINE_ bool operator!=(const Vector3 &p_v) const;
	_FORCE_INLINE_ bool operator<(const Vector3 &p_v) const;
	_FORCE_INLINE_ bool operator<=(const Vector3 &p_v) const;
	_FORCE_INLINE_ bool operator>(const Vector3 &p_v) const;
	_FORCE_INLINE_ bool operator>=(const Vector3 &p_v) const;

	operator String() const;
	_FORCE_INLINE_ operator Vector3i() const {
		return Vector3i(x, y, z);
	}

	_FORCE_INLINE_ Vector3() {}
	_FORCE_INLINE_ Vector3(const Vector3i &p_ivec) {
		x = p_ivec.x;
		y = p_ivec.y;
		z = p_ivec.z;
	}
	_FORCE_INLINE_ Vector3(const real_t p_x, const real_t p_y, const real_t p_z) {
		x = p_x;
		y = p_y;
		z = p_z;
	}
};

Vector3 Vector3::cross(const Vector3 &p_with) const {
	Vector3 ret(
			(y * p_with.z) - (z * p_with.y),
			(z * p_with.x) - (x * p_with.z),
			(x * p_with.y) - (y * p_with.x));

	return ret;
}

real_t Vector3::dot(const Vector3 &p_with) const {
	return x * p_with.x + y * p_with.y + z * p_with.z;
}

Vector3 Vector3::abs() const {
	return Vector3(Math::abs(x), Math::abs(y), Math::abs(z));
}

Vector3 Vector3::sign() const {
	return Vector3(SIGN(x), SIGN(y), SIGN(z));
}

Vector3 Vector3::floor() const {
	return Vector3(Math::floor(x), Math::floor(y), Math::floor(z));
}

Vector3 Vector3::ceil() const {
	return Vector3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
}

Vector3 Vector3::round() const {
	return Vector3(Math::round(x), Math::round(y), Math::round(z));
}

Vector3 Vector3::lerp(const Vector3 &p_to, const real_t p_weight) const {
	return Vector3(
			x + (p_weight * (p_to.x - x)),
			y + (p_weight * (p_to.y - y)),
			z + (p_weight * (p_to.z - z)));
}

Vector3 Vector3::slerp(const Vector3 &p_to, const real_t p_weight) const {
	real_t start_length_sq = length_squared();
	real_t end_length_sq = p_to.length_squared();
	if (unlikely(start_length_sq == 0.0 || end_length_sq == 0.0)) {
		// Zero length vectors have no angle, so the best we can do is either lerp or throw an error.
		return lerp(p_to, p_weight);
	}
	real_t start_length = Math::sqrt(start_length_sq);
	real_t result_length = Math::lerp(start_length, Math::sqrt(end_length_sq), p_weight);
	real_t angle = angle_to(p_to);
	return rotated(cross(p_to).normalized(), angle * p_weight) * (result_length / start_length);
}

real_t Vector3::distance_to(const Vector3 &p_to) const {
	return (p_to - *this).length();
}

real_t Vector3::distance_squared_to(const Vector3 &p_to) const {
	return (p_to - *this).length_squared();
}

Vector3 Vector3::posmod(const real_t p_mod) const {
	return Vector3(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod), Math::fposmod(z, p_mod));
}

Vector3 Vector3::posmodv(const Vector3 &p_modv) const {
	return Vector3(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y), Math::fposmod(z, p_modv.z));
}

Vector3 Vector3::project(const Vector3 &p_to) const {
	return p_to * (dot(p_to) / p_to.length_squared());
}

real_t Vector3::angle_to(const Vector3 &p_to) const {
	return Math::atan2(cross(p_to).length(), dot(p_to));
}

real_t Vector3::signed_angle_to(const Vector3 &p_to, const Vector3 &p_axis) const {
	Vector3 cross_to = cross(p_to);
	real_t unsigned_angle = Math::atan2(cross_to.length(), dot(p_to));
	real_t sign = cross_to.dot(p_axis);
	return (sign < 0) ? -unsigned_angle : unsigned_angle;
}

Vector3 Vector3::direction_to(const Vector3 &p_to) const {
	Vector3 ret(p_to.x - x, p_to.y - y, p_to.z - z);
	ret.normalize();
	return ret;
}

/* Operators */

Vector3 &Vector3::operator+=(const Vector3 &p_v) {
	x += p_v.x;
	y += p_v.y;
	z += p_v.z;
	return *this;
}

Vector3 Vector3::operator+(const Vector3 &p_v) const {
	return Vector3(x + p_v.x, y + p_v.y, z + p_v.z);
}

Vector3 &Vector3::operator-=(const Vector3 &p_v) {
	x -= p_v.x;
	y -= p_v.y;
	z -= p_v.z;
	return *this;
}

Vector3 Vector3::operator-(const Vector3 &p_v) const {
	return Vector3(x - p_v.x, y - p_v.y, z - p_v.z);
}

Vector3 &Vector3::operator*=(const Vector3 &p_v) {
	x *= p_v.x;
	y *= p_v.y;
	z *= p_v.z;
	return *this;
}

Vector3 Vector3::operator*(const Vector3 &p_v) const {
	return Vector3(x * p_v.x, y * p_v.y, z * p_v.z);
}

Vector3 &Vector3::operator/=(const Vector3 &p_v) {
	x /= p_v.x;
	y /= p_v.y;
	z /= p_v.z;
	return *this;
}

Vector3 Vector3::operator/(const Vector3 &p_v) const {
	return Vector3(x / p_v.x, y / p_v.y, z / p_v.z);
}

Vector3 &Vector3::operator*=(const real_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	return *this;
}

_FORCE_INLINE_ Vector3 operator*(const float p_scalar, const Vector3 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(const double p_scalar, const Vector3 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(const int32_t p_scalar, const Vector3 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector3 operator*(const int64_t p_scalar, const Vector3 &p_vec) {
	return p_vec * p_scalar;
}

Vector3 Vector3::operator*(const real_t p_scalar) const {
	return Vector3(x * p_scalar, y * p_scalar, z * p_scalar);
}

Vector3 &Vector3::operator/=(const real_t p_scalar) {
	x /= p_scalar;
	y /= p_scalar;
	z /= p_scalar;
	return *this;
}

Vector3 Vector3::operator/(const real_t p_scalar) const {
	return Vector3(x / p_scalar, y / p_scalar, z / p_scalar);
}

Vector3 Vector3::operator-() const {
	return Vector3(-x, -y, -z);
}

bool Vector3::operator==(const Vector3 &p_v) const {
	return x == p_v.x && y == p_v.y && z == p_v.z;
}

bool Vector3::operator!=(const Vector3 &p_v) const {
	return x != p_v.x || y != p_v.y || z != p_v.z;
}

bool Vector3::operator<(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z < p_v.z;
		}
		return y < p_v.y;
	}
	return x < p_v.x;
}

bool Vector3::operator>(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z > p_v.z;
		}
		return y > p_v.y;
	}
	return x > p_v.x;
}

bool Vector3::operator<=(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z <= p_v.z;
		}
		return y < p_v.y;
	}
	return x < p_v.x;
}

bool Vector3::operator>=(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z >= p_v.z;
		}
		return y > p_v.y;
	}
	return x > p_v.x;
}

_FORCE_INLINE_ Vector3 vec3_cross(const Vector3 &p_a, const Vector3 &p_b) {
	return p_a.cross(p_b);
}

_FORCE_INLINE_ real_t vec3_dot(const Vector3 &p_a, const Vector3 &p_b) {
	return p_a.dot(p_b);
}

real_t Vector3::length() const {
	real_t x2 = x * x;
	real_t y2 = y * y;
	real_t z2 = z * z;

	return Math::sqrt(x2 + y2 + z2);
}

real_t Vector3::length_squared() const {
	real_t x2 = x * x;
	real_t y2 = y * y;
	real_t z2 = z * z;

	return x2 + y2 + z2;
}

void Vector3::normalize() {
	real_t lengthsq = length_squared();
	if (lengthsq == 0) {
		x = y = z = 0;
	} else {
		real_t length = Math::sqrt(lengthsq);
		x /= length;
		y /= length;
		z /= length;
	}
}

Vector3 Vector3::normalized() const {
	Vector3 v = *this;
	v.normalize();
	return v;
}

bool Vector3::is_normalized() const {
	// use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON);
}

Vector3 Vector3::inverse() const {
	return Vector3(1.0 / x, 1.0 / y, 1.0 / z);
}

void Vector3::zero() {
	x = y = z = 0;
}

// slide returns the component of the vector along the given plane, specified by its normal vector.
Vector3 Vector3::slide(const Vector3 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(), "The normal Vector3 must be normalized.");
#endif
	return *this - p_normal * this->dot(p_normal);
}

Vector3 Vector3::bounce(const Vector3 &p_normal) const {
	return -reflect(p_normal);
}

Vector3 Vector3::reflect(const Vector3 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector3(), "The normal Vector3 must be normalized.");
#endif
	return 2.0 * p_normal * this->dot(p_normal) - *this;
}

#endif // VECTOR3_H

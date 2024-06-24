/**************************************************************************/
/*  vector3i.h                                                            */
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

#ifndef VECTOR3I_H
#define VECTOR3I_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"

class String;
struct Vector3;

struct _NO_DISCARD_ Vector3i {
	static const int AXIS_COUNT = 3;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	union {
		struct {
			int32_t x;
			int32_t y;
			int32_t z;
		};

		int32_t coord[3] = { 0 };
	};

	_FORCE_INLINE_ const int32_t &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 3);
		return coord[p_axis];
	}

	_FORCE_INLINE_ int32_t &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 3);
		return coord[p_axis];
	}

	Vector3i::Axis min_axis_index() const;
	Vector3i::Axis max_axis_index() const;

	Vector3i min(const Vector3i &p_vector3i) const {
		return Vector3i(MIN(x, p_vector3i.x), MIN(y, p_vector3i.y), MIN(z, p_vector3i.z));
	}

	Vector3i mini(int32_t p_scalar) const {
		return Vector3i(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
	}

	Vector3i max(const Vector3i &p_vector3i) const {
		return Vector3i(MAX(x, p_vector3i.x), MAX(y, p_vector3i.y), MAX(z, p_vector3i.z));
	}

	Vector3i maxi(int32_t p_scalar) const {
		return Vector3i(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
	}

	_FORCE_INLINE_ int64_t length_squared() const;
	_FORCE_INLINE_ double length() const;

	_FORCE_INLINE_ void zero();

	_FORCE_INLINE_ Vector3i lerp(const Vector3i &p_to, real_t p_weight) const;
	_FORCE_INLINE_ Vector3i slerp(const Vector3i &p_to, real_t p_weight) const;
	_FORCE_INLINE_ Vector3i cubic_interpolate(const Vector3i &p_b, const Vector3i &p_pre_a, const Vector3i &p_post_b, real_t p_weight) const;
	_FORCE_INLINE_ Vector3i cubic_interpolate_in_time(const Vector3i &p_b, const Vector3i &p_pre_a, const Vector3i &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;
	_FORCE_INLINE_ Vector3i bezier_interpolate(const Vector3i &p_control_1, const Vector3i &p_control_2, const Vector3i &p_end, real_t p_t) const;
	_FORCE_INLINE_ Vector3i bezier_derivative(const Vector3i &p_control_1, const Vector3i &p_control_2, const Vector3i &p_end, real_t p_t) const;

	_FORCE_INLINE_ Vector3i cross(const Vector3i &p_with) const;
	_FORCE_INLINE_ int32_t dot(const Vector3i &p_with) const;

	_FORCE_INLINE_ real_t angle_to(const Vector3i &p_to) const;

	bool is_equal(const Vector3i &p_v) const;
	bool is_zero() const;

	void rotate(const Vector3i &p_axis, real_t p_angle);
	Vector3i rotated(const Vector3i &p_axis, real_t p_angle) const;

	_FORCE_INLINE_ Vector3i abs() const;
	_FORCE_INLINE_ Vector3i sign() const;
	Vector3i clamp(const Vector3i &p_min, const Vector3i &p_max) const;
	Vector3i clampi(int32_t p_min, int32_t p_max) const;
	Vector3i snapped(const Vector3i &p_step) const;
	Vector3i snappedi(int32_t p_step) const;

	_FORCE_INLINE_ double distance_to(const Vector3i &p_to) const;
	_FORCE_INLINE_ int64_t distance_squared_to(const Vector3i &p_to) const;

	/* Operators */

	_FORCE_INLINE_ Vector3i &operator+=(const Vector3i &p_v);
	_FORCE_INLINE_ Vector3i operator+(const Vector3i &p_v) const;
	_FORCE_INLINE_ Vector3i &operator-=(const Vector3i &p_v);
	_FORCE_INLINE_ Vector3i operator-(const Vector3i &p_v) const;
	_FORCE_INLINE_ Vector3i &operator*=(const Vector3i &p_v);
	_FORCE_INLINE_ Vector3i operator*(const Vector3i &p_v) const;
	_FORCE_INLINE_ Vector3i &operator/=(const Vector3i &p_v);
	_FORCE_INLINE_ Vector3i operator/(const Vector3i &p_v) const;
	_FORCE_INLINE_ Vector3i &operator%=(const Vector3i &p_v);
	_FORCE_INLINE_ Vector3i operator%(const Vector3i &p_v) const;

	_FORCE_INLINE_ Vector3i &operator*=(int32_t p_scalar);
	_FORCE_INLINE_ Vector3i operator*(int32_t p_scalar) const;
	_FORCE_INLINE_ Vector3i &operator/=(int32_t p_scalar);
	_FORCE_INLINE_ Vector3i operator/(int32_t p_scalar) const;
	_FORCE_INLINE_ Vector3i &operator%=(int32_t p_scalar);
	_FORCE_INLINE_ Vector3i operator%(int32_t p_scalar) const;

	_FORCE_INLINE_ Vector3i operator-() const;

	_FORCE_INLINE_ bool operator==(const Vector3i &p_v) const;
	_FORCE_INLINE_ bool operator!=(const Vector3i &p_v) const;
	_FORCE_INLINE_ bool operator<(const Vector3i &p_v) const;
	_FORCE_INLINE_ bool operator<=(const Vector3i &p_v) const;
	_FORCE_INLINE_ bool operator>(const Vector3i &p_v) const;
	_FORCE_INLINE_ bool operator>=(const Vector3i &p_v) const;

	operator String() const;
	operator Vector3() const;

	_FORCE_INLINE_ Vector3i() {}
	_FORCE_INLINE_ Vector3i(int32_t p_x, int32_t p_y, int32_t p_z) {
		x = p_x;
		y = p_y;
		z = p_z;
	}
};

int64_t Vector3i::length_squared() const {
	return x * (int64_t)x + y * (int64_t)y + z * (int64_t)z;
}

double Vector3i::length() const {
	return Math::sqrt((double)length_squared());
}

Vector3i Vector3i::abs() const {
	return Vector3i(Math::abs(x), Math::abs(y), Math::abs(z));
}

Vector3i Vector3i::sign() const {
	return Vector3i(SIGN(x), SIGN(y), SIGN(z));
}

double Vector3i::distance_to(const Vector3i &p_to) const {
	return (p_to - *this).length();
}

int64_t Vector3i::distance_squared_to(const Vector3i &p_to) const {
	return (p_to - *this).length_squared();
}

/* Operators */

Vector3i &Vector3i::operator+=(const Vector3i &p_v) {
	x += p_v.x;
	y += p_v.y;
	z += p_v.z;
	return *this;
}

Vector3i Vector3i::operator+(const Vector3i &p_v) const {
	return Vector3i(x + p_v.x, y + p_v.y, z + p_v.z);
}

Vector3i &Vector3i::operator-=(const Vector3i &p_v) {
	x -= p_v.x;
	y -= p_v.y;
	z -= p_v.z;
	return *this;
}

Vector3i Vector3i::operator-(const Vector3i &p_v) const {
	return Vector3i(x - p_v.x, y - p_v.y, z - p_v.z);
}

Vector3i &Vector3i::operator*=(const Vector3i &p_v) {
	x *= p_v.x;
	y *= p_v.y;
	z *= p_v.z;
	return *this;
}

Vector3i Vector3i::operator*(const Vector3i &p_v) const {
	return Vector3i(x * p_v.x, y * p_v.y, z * p_v.z);
}

Vector3i &Vector3i::operator/=(const Vector3i &p_v) {
	x /= p_v.x;
	y /= p_v.y;
	z /= p_v.z;
	return *this;
}

Vector3i Vector3i::operator/(const Vector3i &p_v) const {
	return Vector3i(x / p_v.x, y / p_v.y, z / p_v.z);
}

Vector3i &Vector3i::operator%=(const Vector3i &p_v) {
	x %= p_v.x;
	y %= p_v.y;
	z %= p_v.z;
	return *this;
}

Vector3i Vector3i::operator%(const Vector3i &p_v) const {
	return Vector3i(x % p_v.x, y % p_v.y, z % p_v.z);
}

Vector3i &Vector3i::operator*=(int32_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	return *this;
}

Vector3i Vector3i::operator*(int32_t p_scalar) const {
	return Vector3i(x * p_scalar, y * p_scalar, z * p_scalar);
}

// Multiplication operators required to workaround issues with LLVM using implicit conversion.

_FORCE_INLINE_ Vector3i operator*(int32_t p_scalar, const Vector3i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector3i operator*(int64_t p_scalar, const Vector3i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector3i operator*(float p_scalar, const Vector3i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector3i operator*(double p_scalar, const Vector3i &p_vector) {
	return p_vector * p_scalar;
}

Vector3i &Vector3i::operator/=(int32_t p_scalar) {
	x /= p_scalar;
	y /= p_scalar;
	z /= p_scalar;
	return *this;
}

Vector3i Vector3i::operator/(int32_t p_scalar) const {
	return Vector3i(x / p_scalar, y / p_scalar, z / p_scalar);
}

Vector3i &Vector3i::operator%=(int32_t p_scalar) {
	x %= p_scalar;
	y %= p_scalar;
	z %= p_scalar;
	return *this;
}

Vector3i Vector3i::operator%(int32_t p_scalar) const {
	return Vector3i(x % p_scalar, y % p_scalar, z % p_scalar);
}

Vector3i Vector3i::operator-() const {
	return Vector3i(-x, -y, -z);
}

bool Vector3i::operator==(const Vector3i &p_v) const {
	return (x == p_v.x && y == p_v.y && z == p_v.z);
}

bool Vector3i::operator!=(const Vector3i &p_v) const {
	return (x != p_v.x || y != p_v.y || z != p_v.z);
}

bool Vector3i::operator<(const Vector3i &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z < p_v.z;
		} else {
			return y < p_v.y;
		}
	} else {
		return x < p_v.x;
	}
}

bool Vector3i::operator>(const Vector3i &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z > p_v.z;
		} else {
			return y > p_v.y;
		}
	} else {
		return x > p_v.x;
	}
}

bool Vector3i::operator<=(const Vector3i &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z <= p_v.z;
		} else {
			return y < p_v.y;
		}
	} else {
		return x < p_v.x;
	}
}

bool Vector3i::operator>=(const Vector3i &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y) {
			return z >= p_v.z;
		} else {
			return y > p_v.y;
		}
	} else {
		return x > p_v.x;
	}
}

void Vector3i::zero() {
	x = y = z = 0;
}

Vector3i Vector3i::lerp(const Vector3i &p_to, real_t p_weight) const {
	Vector3i res = *this;
	res.x = Math::lerp(res.x, p_to.x, p_weight);
	res.y = Math::lerp(res.y, p_to.y, p_weight);
	res.z = Math::lerp(res.z, p_to.z, p_weight);
	return res;
}

Vector3i Vector3i::slerp(const Vector3i &p_to, real_t p_weight) const {
	// This method seems more complicated than it really is, since we write out
	// the internals of some methods for efficiency (mainly, checking length).
	real_t start_length_sq = length_squared();
	real_t end_length_sq = p_to.length_squared();
	if (unlikely(start_length_sq == 0.0f || end_length_sq == 0.0f)) {
		// Zero length vectors have no angle, so the best we can do is either lerp or throw an error.
		return lerp(p_to, p_weight);
	}
	Vector3i axis = cross(p_to);
	real_t axis_length_sq = axis.length_squared();
	if (unlikely(axis_length_sq == 0.0f)) {
		// Colinear vectors have no rotation axis or angle between them, so the best we can do is lerp.
		return lerp(p_to, p_weight);
	}
	axis /= Math::sqrt(axis_length_sq);
	real_t start_length = Math::sqrt(start_length_sq);
	real_t result_length = Math::lerp(start_length, Math::sqrt(end_length_sq), p_weight);
	real_t angle = angle_to(p_to);
	return rotated(axis, angle * p_weight) * (result_length / start_length);
}

Vector3i Vector3i::cubic_interpolate(const Vector3i &p_b, const Vector3i &p_pre_a, const Vector3i &p_post_b, real_t p_weight) const {
	Vector3i res = *this;
	res.x = Math::cubic_interpolate(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight);
	res.y = Math::cubic_interpolate(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight);
	res.z = Math::cubic_interpolate(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight);
	return res;
}

Vector3i Vector3i::cubic_interpolate_in_time(const Vector3i &p_b, const Vector3i &p_pre_a, const Vector3i &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
	Vector3i res = *this;
	res.x = Math::cubic_interpolate_in_time(res.x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.y = Math::cubic_interpolate_in_time(res.y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	res.z = Math::cubic_interpolate_in_time(res.z, p_b.z, p_pre_a.z, p_post_b.z, p_weight, p_b_t, p_pre_a_t, p_post_b_t);
	return res;
}

Vector3i Vector3i::bezier_interpolate(const Vector3i &p_control_1, const Vector3i &p_control_2, const Vector3i &p_end, real_t p_t) const {
	Vector3i res = *this;
	res.x = Math::bezier_interpolate(res.x, p_control_1.x, p_control_2.x, p_end.x, p_t);
	res.y = Math::bezier_interpolate(res.y, p_control_1.y, p_control_2.y, p_end.y, p_t);
	res.z = Math::bezier_interpolate(res.z, p_control_1.z, p_control_2.z, p_end.z, p_t);
	return res;
}

Vector3i Vector3i::bezier_derivative(const Vector3i &p_control_1, const Vector3i &p_control_2, const Vector3i &p_end, real_t p_t) const {
	Vector3i res = *this;
	res.x = Math::bezier_derivative(res.x, p_control_1.x, p_control_2.x, p_end.x, p_t);
	res.y = Math::bezier_derivative(res.y, p_control_1.y, p_control_2.y, p_end.y, p_t);
	res.z = Math::bezier_derivative(res.z, p_control_1.z, p_control_2.z, p_end.z, p_t);
	return res;
}

Vector3i Vector3i::cross(const Vector3i &p_with) const {
	Vector3i ret(
			(y * p_with.z) - (z * p_with.y),
			(z * p_with.x) - (x * p_with.z),
			(x * p_with.y) - (y * p_with.x));

	return ret;
}

int32_t Vector3i::dot(const Vector3i &p_with) const {
	return x * p_with.x + y * p_with.y + z * p_with.z;
}

real_t Vector3i::angle_to(const Vector3i &p_to) const {
	return Math::atan2(cross(p_to).length(), (double)dot(p_to));
}

#endif // VECTOR3I_H

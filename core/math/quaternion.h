/**************************************************************************/
/*  quaternion.h                                                          */
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
#include "core/math/vector3.h"

struct [[nodiscard]] Quaternion {
	union {
		// NOLINTBEGIN(modernize-use-default-member-init)
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t components[4] = { 0, 0, 0, 1.0 };
		// NOLINTEND(modernize-use-default-member-init)
	};

	_FORCE_INLINE_ real_t &operator[](int p_idx) {
		return components[p_idx];
	}
	_FORCE_INLINE_ const real_t &operator[](int p_idx) const {
		return components[p_idx];
	}
	_FORCE_INLINE_ real_t length_squared() const;
	_FORCE_INLINE_ bool is_equal_approx(const Quaternion &p_quaternion) const;
	_FORCE_INLINE_ bool is_same(const Quaternion &p_quaternion) const;
	_FORCE_INLINE_ bool is_finite() const;
	_FORCE_INLINE_ real_t length() const;
	_FORCE_INLINE_ void normalize();
	_FORCE_INLINE_ Quaternion normalized() const;
	_FORCE_INLINE_ bool is_normalized() const;
	Quaternion inverse() const;
	_FORCE_INLINE_ Quaternion log() const;
	_FORCE_INLINE_ Quaternion exp() const;
	_FORCE_INLINE_ real_t dot(const Quaternion &p_q) const;
	_FORCE_INLINE_ real_t angle_to(const Quaternion &p_to) const;

	Vector3 get_euler(EulerOrder p_order = EulerOrder::YXZ) const;
	static _FORCE_INLINE_ Quaternion from_euler(const Vector3 &p_euler);

	Quaternion slerp(const Quaternion &p_to, real_t p_weight) const;
	Quaternion slerpni(const Quaternion &p_to, real_t p_weight) const;
	Quaternion spherical_cubic_interpolate(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight) const;
	Quaternion spherical_cubic_interpolate_in_time(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

	_FORCE_INLINE_ Vector3 get_axis() const;
	_FORCE_INLINE_ real_t get_angle() const;

	_FORCE_INLINE_ void get_axis_angle(Vector3 &r_axis, real_t &r_angle) const {
		r_angle = 2 * Math::acos(w);
		real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
		r_axis.x = x * r;
		r_axis.y = y * r;
		r_axis.z = z * r;
	}

	_FORCE_INLINE_ constexpr void operator*=(const Quaternion &p_q);
	_FORCE_INLINE_ constexpr Quaternion operator*(const Quaternion &p_q) const;

	Vector3 xform(const Vector3 &p_v) const;
	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_v) const {
		return inverse().xform(p_v);
	}

	_FORCE_INLINE_ constexpr void operator+=(const Quaternion &p_q);
	_FORCE_INLINE_ constexpr void operator-=(const Quaternion &p_q);
	_FORCE_INLINE_ constexpr void operator*=(real_t p_s);
	_FORCE_INLINE_ constexpr void operator/=(real_t p_s);
	_FORCE_INLINE_ constexpr Quaternion operator+(const Quaternion &p_q2) const;
	_FORCE_INLINE_ constexpr Quaternion operator-(const Quaternion &p_q2) const;
	_FORCE_INLINE_ constexpr Quaternion operator-() const;
	_FORCE_INLINE_ constexpr Quaternion operator*(real_t p_s) const;
	_FORCE_INLINE_ constexpr Quaternion operator/(real_t p_s) const;

	_FORCE_INLINE_ constexpr bool operator==(const Quaternion &p_quaternion) const;
	_FORCE_INLINE_ constexpr bool operator!=(const Quaternion &p_quaternion) const;

	operator String() const;

	_FORCE_INLINE_ constexpr Quaternion() :
			x(0), y(0), z(0), w(1) {}

	_FORCE_INLINE_ constexpr Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x), y(p_y), z(p_z), w(p_w) {}

	Quaternion(const Vector3 &p_axis, real_t p_angle);

	_FORCE_INLINE_ constexpr Quaternion(const Quaternion &p_q) :
			x(p_q.x), y(p_q.y), z(p_q.z), w(p_q.w) {}

	_FORCE_INLINE_ constexpr void operator=(const Quaternion &p_q) {
		x = p_q.x;
		y = p_q.y;
		z = p_q.z;
		w = p_q.w;
	}

	Quaternion(const Vector3 &p_v0, const Vector3 &p_v1) { // Shortest arc.
#ifdef MATH_CHECKS
		ERR_FAIL_COND_MSG(p_v0.is_zero_approx() || p_v1.is_zero_approx(), "The vectors must not be zero.");
#endif
		constexpr real_t ALMOST_ONE = 1.0f - (real_t)CMP_EPSILON;
		Vector3 n0 = p_v0.normalized();
		Vector3 n1 = p_v1.normalized();
		real_t d = n0.dot(n1);
		if (Math::abs(d) > ALMOST_ONE) {
			if (d >= 0) {
				return; // Vectors are same.
			}
			Vector3 axis = n0.get_any_perpendicular();
			x = axis.x;
			y = axis.y;
			z = axis.z;
			w = 0;
		} else {
			Vector3 c = n0.cross(n1);
			real_t s = Math::sqrt((1.0f + d) * 2.0f);
			real_t rs = 1.0f / s;

			x = c.x * rs;
			y = c.y * rs;
			z = c.z * rs;
			w = s * 0.5f;
		}
	}
};

real_t Quaternion::dot(const Quaternion &p_q) const {
	return x * p_q.x + y * p_q.y + z * p_q.z + w * p_q.w;
}

real_t Quaternion::length_squared() const {
	return dot(*this);
}

constexpr void Quaternion::operator+=(const Quaternion &p_q) {
	x += p_q.x;
	y += p_q.y;
	z += p_q.z;
	w += p_q.w;
}

constexpr void Quaternion::operator-=(const Quaternion &p_q) {
	x -= p_q.x;
	y -= p_q.y;
	z -= p_q.z;
	w -= p_q.w;
}

constexpr void Quaternion::operator*=(real_t p_s) {
	x *= p_s;
	y *= p_s;
	z *= p_s;
	w *= p_s;
}

constexpr void Quaternion::operator/=(real_t p_s) {
	*this *= (1 / p_s);
}

constexpr Quaternion Quaternion::operator+(const Quaternion &p_q2) const {
	const Quaternion &q1 = *this;
	return Quaternion(q1.x + p_q2.x, q1.y + p_q2.y, q1.z + p_q2.z, q1.w + p_q2.w);
}

constexpr Quaternion Quaternion::operator-(const Quaternion &p_q2) const {
	const Quaternion &q1 = *this;
	return Quaternion(q1.x - p_q2.x, q1.y - p_q2.y, q1.z - p_q2.z, q1.w - p_q2.w);
}

constexpr Quaternion Quaternion::operator-() const {
	const Quaternion &q2 = *this;
	return Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
}

constexpr Quaternion Quaternion::operator*(real_t p_s) const {
	return Quaternion(x * p_s, y * p_s, z * p_s, w * p_s);
}

constexpr Quaternion Quaternion::operator/(real_t p_s) const {
	return *this * (1 / p_s);
}

constexpr bool Quaternion::operator==(const Quaternion &p_quaternion) const {
	return x == p_quaternion.x && y == p_quaternion.y && z == p_quaternion.z && w == p_quaternion.w;
}

constexpr bool Quaternion::operator!=(const Quaternion &p_quaternion) const {
	return x != p_quaternion.x || y != p_quaternion.y || z != p_quaternion.z || w != p_quaternion.w;
}

constexpr void Quaternion::operator*=(const Quaternion &p_q) {
	real_t xx = w * p_q.x + x * p_q.w + y * p_q.z - z * p_q.y;
	real_t yy = w * p_q.y + y * p_q.w + z * p_q.x - x * p_q.z;
	real_t zz = w * p_q.z + z * p_q.w + x * p_q.y - y * p_q.x;
	w = w * p_q.w - x * p_q.x - y * p_q.y - z * p_q.z;
	x = xx;
	y = yy;
	z = zz;
}

constexpr Quaternion Quaternion::operator*(const Quaternion &p_q) const {
	Quaternion r = *this;
	r *= p_q;
	return r;
}

_FORCE_INLINE_ constexpr Quaternion operator*(real_t p_real, const Quaternion &p_quaternion) {
	return p_quaternion * p_real;
}

real_t Quaternion::angle_to(const Quaternion &p_to) const {
	real_t d = dot(p_to);
	// acos does clamping.
	return Math::acos(d * d * 2 - 1);
}

bool Quaternion::is_equal_approx(const Quaternion &p_quaternion) const {
	return Math::is_equal_approx(x, p_quaternion.x) && Math::is_equal_approx(y, p_quaternion.y) && Math::is_equal_approx(z, p_quaternion.z) && Math::is_equal_approx(w, p_quaternion.w);
}

bool Quaternion::is_same(const Quaternion &p_quaternion) const {
	return Math::is_same(x, p_quaternion.x) && Math::is_same(y, p_quaternion.y) && Math::is_same(z, p_quaternion.z) && Math::is_same(w, p_quaternion.w);
}

bool Quaternion::is_finite() const {
	return Math::is_finite(x) && Math::is_finite(y) && Math::is_finite(z) && Math::is_finite(w);
}

real_t Quaternion::length() const {
	return Math::sqrt(length_squared());
}

void Quaternion::normalize() {
	*this /= length();
}

Quaternion Quaternion::normalized() const {
	return *this / length();
}

bool Quaternion::is_normalized() const {
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON); //use less epsilon
}

Quaternion Quaternion::log() const {
	Quaternion src = *this;
	Vector3 src_v = src.get_axis() * src.get_angle();
	return Quaternion(src_v.x, src_v.y, src_v.z, 0);
}

Quaternion Quaternion::exp() const {
	Quaternion src = *this;
	Vector3 src_v = Vector3(src.x, src.y, src.z);
	real_t theta = src_v.length();
	src_v = src_v.normalized();
	if (theta < CMP_EPSILON || !src_v.is_normalized()) {
		return Quaternion(0, 0, 0, 1);
	}
	return Quaternion(src_v, theta);
}

Vector3 Quaternion::get_axis() const {
	if (Math::abs(w) > 1 - CMP_EPSILON) {
		return Vector3(x, y, z);
	}
	real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
	return Vector3(x * r, y * r, z * r);
}

real_t Quaternion::get_angle() const {
	return 2 * Math::acos(w);
}

// Euler constructor expects a vector containing the Euler angles in the format
// (ax, ay, az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
Quaternion Quaternion::from_euler(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.y * 0.5f;
	real_t half_a2 = p_euler.x * 0.5f;
	real_t half_a3 = p_euler.z * 0.5f;

	// R = Y(a1).X(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = Math::cos(half_a1);
	real_t sin_a1 = Math::sin(half_a1);
	real_t cos_a2 = Math::cos(half_a2);
	real_t sin_a2 = Math::sin(half_a2);
	real_t cos_a3 = Math::cos(half_a3);
	real_t sin_a3 = Math::sin(half_a3);

	return Quaternion(
			sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3,
			sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3,
			-sin_a1 * sin_a2 * cos_a3 + cos_a1 * cos_a2 * sin_a3,
			sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

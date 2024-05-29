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

#ifndef QUATERNION_H
#define QUATERNION_H

#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/string/ustring.h"

struct _NO_DISCARD_ Quaternion {
	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t components[4] = { 0, 0, 0, 1 };
	};

	constexpr const real_t &operator[](size_t p_idx) const;
	constexpr real_t &operator[](size_t p_idx);

	_FORCE_INLINE_ real_t length_squared() const;
	bool is_equal_approx(const Quaternion &p_quaternion) const;
	bool is_finite() const;
	real_t length() const;
	void normalize();
	Quaternion normalized() const;
	bool is_normalized() const;
	Quaternion inverse() const;
	Quaternion log() const;
	Quaternion exp() const;
	_FORCE_INLINE_ real_t dot(const Quaternion &p_q) const;
	real_t angle_to(const Quaternion &p_to) const;

	Vector3 get_euler(EulerOrder p_order = EulerOrder::YXZ) const;
	static Quaternion from_euler(const Vector3 &p_euler);

	Quaternion slerp(const Quaternion &p_to, real_t p_weight) const;
	Quaternion slerpni(const Quaternion &p_to, real_t p_weight) const;
	Quaternion spherical_cubic_interpolate(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight) const;
	Quaternion spherical_cubic_interpolate_in_time(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const;

	Vector3 get_axis() const;
	real_t get_angle() const;

	_FORCE_INLINE_ void get_axis_angle(Vector3 &r_axis, real_t &r_angle) const {
		r_angle = 2 * Math::acos(w);
		real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
		r_axis.x = x * r;
		r_axis.y = y * r;
		r_axis.z = z * r;
	}

	_FORCE_INLINE_ Vector3 xform(const Vector3 &p_v) const {
#ifdef MATH_CHECKS
		ERR_FAIL_COND_V_MSG(!is_normalized(), p_v, "The quaternion " + operator String() + " must be normalized.");
#endif
		Vector3 u(x, y, z);
		Vector3 uv = u.cross(p_v);
		return p_v + ((uv * w) + u.cross(uv)) * ((real_t)2);
	}

	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_v) const {
		return inverse().xform(p_v);
	}

	constexpr Quaternion &operator+=(const Quaternion &p_other);
	constexpr Quaternion operator+(const Quaternion &p_other) const;
	constexpr Quaternion &operator-=(const Quaternion &p_other);
	constexpr Quaternion operator-(const Quaternion &p_other) const;
	constexpr Quaternion &operator*=(const Quaternion &p_other);
	constexpr Quaternion operator*(const Quaternion &p_other) const;

	constexpr Quaternion &operator*=(real_t p_scalar);
	constexpr Quaternion operator*(real_t p_scalar) const;
	constexpr Quaternion &operator/=(real_t p_scalar);
	constexpr Quaternion operator/(real_t p_scalar) const;

	constexpr Quaternion operator-() const;

	constexpr bool operator==(const Quaternion &p_other) const;
	constexpr bool operator!=(const Quaternion &p_other) const;

	operator String() const;

	constexpr Quaternion() :
			x(0), y(0), z(0), w(1) {}
	constexpr Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x), y(p_y), z(p_z), w(p_w) {}

	Quaternion(const Vector3 &p_axis, real_t p_angle);

	Quaternion(const Vector3 &p_v0, const Vector3 &p_v1) { // Shortest arc.
		Vector3 c = p_v0.cross(p_v1);
		real_t d = p_v0.dot(p_v1);

		if (d < -1.0f + (real_t)CMP_EPSILON) {
			x = 0;
			y = 1;
			z = 0;
			w = 0;
		} else {
			real_t s = Math::sqrt((1.0f + d) * 2.0f);
			real_t rs = 1.0f / s;

			x = c.x * rs;
			y = c.y * rs;
			z = c.z * rs;
			w = s * 0.5f;
		}
	}
};

constexpr const real_t &Quaternion::operator[](size_t p_idx) const {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_idx, 4);
	}
#endif
	switch (p_idx) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		default:
			return components[p_idx];
	}
}

constexpr real_t &Quaternion::operator[](size_t p_idx) {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_idx, 4);
	}
#endif
	switch (p_idx) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		default:
			return components[p_idx];
	}
}

constexpr Quaternion &Quaternion::operator+=(const Quaternion &p_other) {
	x += p_other.x;
	y += p_other.y;
	z += p_other.z;
	w += p_other.w;
	return *this;
}

constexpr Quaternion Quaternion::operator+(const Quaternion &p_other) const {
	return Quaternion(x + p_other.x, y + p_other.y, z + p_other.z, w + p_other.w);
}

constexpr Quaternion &Quaternion::operator-=(const Quaternion &p_other) {
	x -= p_other.x;
	y -= p_other.y;
	z -= p_other.z;
	w -= p_other.w;
	return *this;
}

constexpr Quaternion Quaternion::operator-(const Quaternion &p_other) const {
	return Quaternion(x - p_other.x, y - p_other.y, z - p_other.z, w - p_other.w);
}

constexpr Quaternion &Quaternion::operator*=(const Quaternion &p_other) {
	real_t xx = w * p_other.x + x * p_other.w + y * p_other.z - z * p_other.y;
	real_t yy = w * p_other.y + y * p_other.w + z * p_other.x - x * p_other.z;
	real_t zz = w * p_other.z + z * p_other.w + x * p_other.y - y * p_other.x;
	w = w * p_other.w - x * p_other.x - y * p_other.y - z * p_other.z;
	x = xx;
	y = yy;
	z = zz;
	return *this;
}

constexpr Quaternion Quaternion::operator*(const Quaternion &p_other) const {
	Quaternion r = *this;
	r *= p_other;
	return r;
}

constexpr Quaternion &Quaternion::operator*=(real_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	w *= p_scalar;
	return *this;
}

constexpr Quaternion Quaternion::operator*(real_t p_scalar) const {
	return Quaternion(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
}

constexpr Quaternion &Quaternion::operator/=(real_t p_scalar) {
	*this *= 1 / p_scalar;
	return *this;
}

constexpr Quaternion Quaternion::operator/(real_t p_scalar) const {
	return *this * (1 / p_scalar);
}

constexpr Quaternion Quaternion::operator-() const {
	return Quaternion(-x, -y, -z, -w);
}

constexpr bool Quaternion::operator==(const Quaternion &p_other) const {
	return x == p_other.x && y == p_other.y && z == p_other.z && w == p_other.w;
}

constexpr bool Quaternion::operator!=(const Quaternion &p_other) const {
	return x != p_other.x || y != p_other.y || z != p_other.z || w != p_other.w;
}

constexpr Quaternion operator*(real_t p_real, const Quaternion &p_other) {
	return p_other * p_real;
}

real_t Quaternion::dot(const Quaternion &p_q) const {
	return x * p_q.x + y * p_q.y + z * p_q.z + w * p_q.w;
}

real_t Quaternion::length_squared() const {
	return dot(*this);
}

#endif // QUATERNION_H

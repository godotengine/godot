/*************************************************************************/
/*  quaternion.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef QUATERNION_H
#define QUATERNION_H

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/string/ustring.h"

class Quaternion {
public:
	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
			real_t w;
		};
		real_t components[4] = { 0, 0, 0, 1.0 };
	};

	_FORCE_INLINE_ real_t &operator[](int idx) {
		return components[idx];
	}
	_FORCE_INLINE_ const real_t &operator[](int idx) const {
		return components[idx];
	}
	_FORCE_INLINE_ real_t length_squared() const;
	bool is_equal_approx(const Quaternion &p_quaternion) const;
	real_t length() const;
	void normalize();
	Quaternion normalized() const;
	bool is_normalized() const;
	Quaternion inverse() const;
	_FORCE_INLINE_ real_t dot(const Quaternion &p_q) const;
	real_t angle_to(const Quaternion &p_to) const;

	Vector3 get_euler_xyz() const;
	Vector3 get_euler_yxz() const;
	Vector3 get_euler() const { return get_euler_yxz(); };

	Quaternion slerp(const Quaternion &p_to, const real_t &p_weight) const;
	Quaternion slerpni(const Quaternion &p_to, const real_t &p_weight) const;
	Quaternion cubic_slerp(const Quaternion &p_b, const Quaternion &p_pre_a, const Quaternion &p_post_b, const real_t &p_weight) const;

	Vector3 get_axis() const;
	float get_angle() const;

	_FORCE_INLINE_ void get_axis_angle(Vector3 &r_axis, real_t &r_angle) const {
		r_angle = 2 * Math::acos(w);
		real_t r = ((real_t)1) / Math::sqrt(1 - w * w);
		r_axis.x = x * r;
		r_axis.y = y * r;
		r_axis.z = z * r;
	}

	void operator*=(const Quaternion &p_q);
	Quaternion operator*(const Quaternion &p_q) const;

	_FORCE_INLINE_ Vector3 xform(const Vector3 &v) const {
#ifdef MATH_CHECKS
		ERR_FAIL_COND_V_MSG(!is_normalized(), v, "The quaternion must be normalized.");
#endif
		Vector3 u(x, y, z);
		Vector3 uv = u.cross(v);
		return v + ((uv * w) + u.cross(uv)) * ((real_t)2);
	}

	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &v) const {
		return inverse().xform(v);
	}

	_FORCE_INLINE_ void operator+=(const Quaternion &p_q);
	_FORCE_INLINE_ void operator-=(const Quaternion &p_q);
	_FORCE_INLINE_ void operator*=(const real_t &s);
	_FORCE_INLINE_ void operator/=(const real_t &s);
	_FORCE_INLINE_ Quaternion operator+(const Quaternion &q2) const;
	_FORCE_INLINE_ Quaternion operator-(const Quaternion &q2) const;
	_FORCE_INLINE_ Quaternion operator-() const;
	_FORCE_INLINE_ Quaternion operator*(const real_t &s) const;
	_FORCE_INLINE_ Quaternion operator/(const real_t &s) const;

	_FORCE_INLINE_ bool operator==(const Quaternion &p_quaternion) const;
	_FORCE_INLINE_ bool operator!=(const Quaternion &p_quaternion) const;

	operator String() const;

	_FORCE_INLINE_ Quaternion() {}

	_FORCE_INLINE_ Quaternion(real_t p_x, real_t p_y, real_t p_z, real_t p_w) :
			x(p_x),
			y(p_y),
			z(p_z),
			w(p_w) {
	}

	Quaternion(const Vector3 &p_axis, real_t p_angle);

	Quaternion(const Vector3 &p_euler);

	Quaternion(const Quaternion &p_q) :
			x(p_q.x),
			y(p_q.y),
			z(p_q.z),
			w(p_q.w) {
	}

	void operator=(const Quaternion &p_q) {
		x = p_q.x;
		y = p_q.y;
		z = p_q.z;
		w = p_q.w;
	}

	Quaternion(const Vector3 &v0, const Vector3 &v1) // shortest arc
	{
		Vector3 c = v0.cross(v1);
		real_t d = v0.dot(v1);

		if (d < -1.0 + CMP_EPSILON) {
			x = 0;
			y = 1;
			z = 0;
			w = 0;
		} else {
			real_t s = Math::sqrt((1.0 + d) * 2.0);
			real_t rs = 1.0 / s;

			x = c.x * rs;
			y = c.y * rs;
			z = c.z * rs;
			w = s * 0.5;
		}
	}
};

real_t Quaternion::dot(const Quaternion &p_q) const {
	return x * p_q.x + y * p_q.y + z * p_q.z + w * p_q.w;
}

real_t Quaternion::length_squared() const {
	return dot(*this);
}

void Quaternion::operator+=(const Quaternion &p_q) {
	x += p_q.x;
	y += p_q.y;
	z += p_q.z;
	w += p_q.w;
}

void Quaternion::operator-=(const Quaternion &p_q) {
	x -= p_q.x;
	y -= p_q.y;
	z -= p_q.z;
	w -= p_q.w;
}

void Quaternion::operator*=(const real_t &s) {
	x *= s;
	y *= s;
	z *= s;
	w *= s;
}

void Quaternion::operator/=(const real_t &s) {
	*this *= 1.0 / s;
}

Quaternion Quaternion::operator+(const Quaternion &q2) const {
	const Quaternion &q1 = *this;
	return Quaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
}

Quaternion Quaternion::operator-(const Quaternion &q2) const {
	const Quaternion &q1 = *this;
	return Quaternion(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
}

Quaternion Quaternion::operator-() const {
	const Quaternion &q2 = *this;
	return Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
}

Quaternion Quaternion::operator*(const real_t &s) const {
	return Quaternion(x * s, y * s, z * s, w * s);
}

Quaternion Quaternion::operator/(const real_t &s) const {
	return *this * (1.0 / s);
}

bool Quaternion::operator==(const Quaternion &p_quaternion) const {
	return x == p_quaternion.x && y == p_quaternion.y && z == p_quaternion.z && w == p_quaternion.w;
}

bool Quaternion::operator!=(const Quaternion &p_quaternion) const {
	return x != p_quaternion.x || y != p_quaternion.y || z != p_quaternion.z || w != p_quaternion.w;
}

_FORCE_INLINE_ Quaternion operator*(const real_t &p_real, const Quaternion &p_quaternion) {
	return p_quaternion * p_real;
}

#endif // QUATERNION_H

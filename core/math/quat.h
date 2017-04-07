/*************************************************************************/
/*  quat.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "vector3.h"

#ifndef QUAT_H
#define QUAT_H

#include "math_defs.h"
#include "math_funcs.h"
#include "ustring.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Quat {
public:
	real_t x, y, z, w;

	_FORCE_INLINE_ real_t length_squared() const;
	real_t length() const;
	void normalize();
	Quat normalized() const;
	Quat inverse() const;
	_FORCE_INLINE_ real_t dot(const Quat &q) const;
	void set_euler(const Vector3 &p_euler);
	Vector3 get_euler() const;
	Quat slerp(const Quat &q, const real_t &t) const;
	Quat slerpni(const Quat &q, const real_t &t) const;
	Quat cubic_slerp(const Quat &q, const Quat &prep, const Quat &postq, const real_t &t) const;

	_FORCE_INLINE_ void get_axis_and_angle(Vector3 &r_axis, real_t &r_angle) const {
		r_angle = 2 * Math::acos(w);
		r_axis.x = x / Math::sqrt(1 - w * w);
		r_axis.y = y / Math::sqrt(1 - w * w);
		r_axis.z = z / Math::sqrt(1 - w * w);
	}

	void operator*=(const Quat &q);
	Quat operator*(const Quat &q) const;

	Quat operator*(const Vector3 &v) const {
		return Quat(w * v.x + y * v.z - z * v.y,
				w * v.y + z * v.x - x * v.z,
				w * v.z + x * v.y - y * v.x,
				-x * v.x - y * v.y - z * v.z);
	}

	_FORCE_INLINE_ Vector3 xform(const Vector3 &v) const {

		Quat q = *this * v;
		q *= this->inverse();
		return Vector3(q.x, q.y, q.z);
	}

	_FORCE_INLINE_ void operator+=(const Quat &q);
	_FORCE_INLINE_ void operator-=(const Quat &q);
	_FORCE_INLINE_ void operator*=(const real_t &s);
	_FORCE_INLINE_ void operator/=(const real_t &s);
	_FORCE_INLINE_ Quat operator+(const Quat &q2) const;
	_FORCE_INLINE_ Quat operator-(const Quat &q2) const;
	_FORCE_INLINE_ Quat operator-() const;
	_FORCE_INLINE_ Quat operator*(const real_t &s) const;
	_FORCE_INLINE_ Quat operator/(const real_t &s) const;

	_FORCE_INLINE_ bool operator==(const Quat &p_quat) const;
	_FORCE_INLINE_ bool operator!=(const Quat &p_quat) const;

	operator String() const;

	inline void set(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
		x = p_x;
		y = p_y;
		z = p_z;
		w = p_w;
	}
	inline Quat(real_t p_x, real_t p_y, real_t p_z, real_t p_w) {
		x = p_x;
		y = p_y;
		z = p_z;
		w = p_w;
	}
	Quat(const Vector3 &axis, const real_t &angle);

	Quat(const Vector3 &v0, const Vector3 &v1) // shortest arc
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

	inline Quat() {
		x = y = z = 0;
		w = 1;
	}
};

real_t Quat::dot(const Quat &q) const {
	return x * q.x + y * q.y + z * q.z + w * q.w;
}

real_t Quat::length_squared() const {
	return dot(*this);
}

void Quat::operator+=(const Quat &q) {
	x += q.x;
	y += q.y;
	z += q.z;
	w += q.w;
}

void Quat::operator-=(const Quat &q) {
	x -= q.x;
	y -= q.y;
	z -= q.z;
	w -= q.w;
}

void Quat::operator*=(const real_t &s) {
	x *= s;
	y *= s;
	z *= s;
	w *= s;
}

void Quat::operator/=(const real_t &s) {

	*this *= 1.0 / s;
}

Quat Quat::operator+(const Quat &q2) const {
	const Quat &q1 = *this;
	return Quat(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
}

Quat Quat::operator-(const Quat &q2) const {
	const Quat &q1 = *this;
	return Quat(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
}

Quat Quat::operator-() const {
	const Quat &q2 = *this;
	return Quat(-q2.x, -q2.y, -q2.z, -q2.w);
}

Quat Quat::operator*(const real_t &s) const {
	return Quat(x * s, y * s, z * s, w * s);
}

Quat Quat::operator/(const real_t &s) const {
	return *this * (1.0 / s);
}

bool Quat::operator==(const Quat &p_quat) const {
	return x == p_quat.x && y == p_quat.y && z == p_quat.z && w == p_quat.w;
}

bool Quat::operator!=(const Quat &p_quat) const {
	return x != p_quat.x || y != p_quat.y || z != p_quat.z || w != p_quat.w;
}

#endif

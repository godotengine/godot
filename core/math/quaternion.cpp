/*************************************************************************/
/*  quaternion.cpp                                                       */
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

#include "quaternion.h"

#include "core/math/basis.h"
#include "core/string/print_string.h"

// get_euler_xyz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses XYZ convention (Z is the first rotation).
Vector3 Quaternion::get_euler_xyz() const {
	Basis m(*this);
	return m.get_euler_xyz();
}

// get_euler_yxz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
Vector3 Quaternion::get_euler_yxz() const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Vector3(0, 0, 0), "The quaternion must be normalized.");
#endif
	Basis m(*this);
	return m.get_euler_yxz();
}

void Quaternion::operator*=(const Quaternion &p_q) {
	real_t xx = w * p_q.x + x * p_q.w + y * p_q.z - z * p_q.y;
	real_t yy = w * p_q.y + y * p_q.w + z * p_q.x - x * p_q.z;
	real_t zz = w * p_q.z + z * p_q.w + x * p_q.y - y * p_q.x;
	w = w * p_q.w - x * p_q.x - y * p_q.y - z * p_q.z;
	x = xx;
	y = yy;
	z = zz;
}

Quaternion Quaternion::operator*(const Quaternion &p_q) const {
	Quaternion r = *this;
	r *= p_q;
	return r;
}

bool Quaternion::is_equal_approx(const Quaternion &p_quaternion) const {
	return Math::is_equal_approx(x, p_quaternion.x) && Math::is_equal_approx(y, p_quaternion.y) && Math::is_equal_approx(z, p_quaternion.z) && Math::is_equal_approx(w, p_quaternion.w);
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

Quaternion Quaternion::inverse() const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The quaternion must be normalized.");
#endif
	return Quaternion(-x, -y, -z, w);
}

Quaternion Quaternion::slerp(const Quaternion &p_to, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	Quaternion to1;
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = dot(p_to);

	// adjust signs (if necessary)
	if (cosom < 0.0) {
		cosom = -cosom;
		to1.x = -p_to.x;
		to1.y = -p_to.y;
		to1.z = -p_to.z;
		to1.w = -p_to.w;
	} else {
		to1.x = p_to.x;
		to1.y = p_to.y;
		to1.z = p_to.z;
		to1.w = p_to.w;
	}

	// calculate coefficients

	if ((1.0 - cosom) > CMP_EPSILON) {
		// standard case (slerp)
		omega = Math::acos(cosom);
		sinom = Math::sin(omega);
		scale0 = Math::sin((1.0 - p_weight) * omega) / sinom;
		scale1 = Math::sin(p_weight * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1.0 - p_weight;
		scale1 = p_weight;
	}
	// calculate final values
	return Quaternion(
			scale0 * x + scale1 * to1.x,
			scale0 * y + scale1 * to1.y,
			scale0 * z + scale1 * to1.z,
			scale0 * w + scale1 * to1.w);
}

Quaternion Quaternion::slerpni(const Quaternion &p_to, const real_t &p_weight) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_to.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	const Quaternion &from = *this;

	real_t dot = from.dot(p_to);

	if (Math::absf(dot) > 0.9999) {
		return from;
	}

	real_t theta = Math::acos(dot),
		   sinT = 1.0 / Math::sin(theta),
		   newFactor = Math::sin(p_weight * theta) * sinT,
		   invFactor = Math::sin((1.0 - p_weight) * theta) * sinT;

	return Quaternion(invFactor * from.x + newFactor * p_to.x,
			invFactor * from.y + newFactor * p_to.y,
			invFactor * from.z + newFactor * p_to.z,
			invFactor * from.w + newFactor * p_to.w);
}

Quaternion Quaternion::cubic_slerp(const Quaternion &p_q, const Quaternion &p_prep, const Quaternion &p_postq, const real_t &p_t) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Quaternion(), "The start quaternion must be normalized.");
	ERR_FAIL_COND_V_MSG(!p_q.is_normalized(), Quaternion(), "The end quaternion must be normalized.");
#endif
	// Modify quaternions for shortest path
	// https://math.stackexchange.com/questions/2650188/super-confused-by-squad-algorithm-for-quaternion-interpolation
	const Quaternion q_a = *this;
	Quaternion prep = (q_a - p_prep).length_squared() < (q_a + p_prep).length_squared() ? p_prep : p_prep * -1.0f;
	Quaternion q_b = (q_a - p_q).length_squared() < (q_a + p_q).length_squared() ? p_q : p_q * -1.0f;
	Quaternion postq = (p_q - p_postq).length_squared() < (p_q + p_postq).length_squared() ? p_postq : p_postq * -1.0f;

	return prep.spline_segment(q_a, q_b, postq, p_t);
}

Quaternion Quaternion::squad(const Quaternion p_a, const Quaternion p_b, const Quaternion p_post, const float p_t) const {
	Quaternion pre = *this;
	float slerp_t = 2.0 * p_t * (1.0 - p_t);
	Quaternion slerp_1 = pre.slerpni(p_post, p_t);
	Quaternion slerp_2 = p_a.slerpni(p_b, p_t);
	return slerp_1.slerpni(slerp_2, slerp_t);
}

Quaternion Quaternion::log() const {
	Quaternion result = *this;
	float a_0 = result.w;
	result.w = 0.0;
	if (Math::abs(a_0) < 1.0) {
		float angle = Math::acos(a_0);
		angle = CLAMP(angle, -1.0f, 1.0f);
		float sin_angle = Math::sin(angle);
		if (!Math::is_equal_approx(Math::absf(sin_angle), 0.0f)) {
			float coeff = angle / sin_angle;
			result.x *= coeff;
			result.y *= coeff;
			result.z *= coeff;
		}
	}
	return result;
}

Quaternion Quaternion::exp() const {
	Quaternion rot = *this;

	float angle = rot.length();
	float coeff = 0.0f;

	if (!Math::is_equal_approx(angle, 0.0f)) {
		coeff = Math::sin(angle) / angle;
	}
	Quaternion result;
	result.x = rot.x * coeff;
	result.y = rot.y * coeff;
	result.z = rot.z * coeff;
	result.w = Math::cos(angle);
	return result;
}

Quaternion Quaternion::intermediate(Quaternion p_a, Quaternion p_b) const {
	Quaternion a_inv = p_a.inverse();
	Quaternion c_1 = a_inv * p_b;
	Quaternion c_2 = a_inv * (*this);
	c_1 = c_1.log();
	c_2 = c_2.log();
	Quaternion c_3 = c_2 + c_1;
	c_3 = c_3 * -0.25f;
	c_3 = c_3.exp();
	Quaternion r = p_a * c_3;
	return r.normalized();
}

Quaternion Quaternion::spline_segment(const Quaternion p_a, const Quaternion p_b, const Quaternion p_post, const float p_t) const {
	Quaternion pre = *this;
	Quaternion q_a = pre.intermediate(p_a, p_b);
	Quaternion q_b = p_a.intermediate(p_b, p_post);
	return p_a.squad(q_a, q_b, p_b, p_t);
}

Quaternion::operator String() const {
	return "(" + String::num_real(x, false) + ", " + String::num_real(y, false) + ", " + String::num_real(z, false) + ", " + String::num_real(w, false) + ")";
}

Quaternion::Quaternion(const Vector3 &p_axis, real_t p_angle) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_MSG(!p_axis.is_normalized(), "The axis Vector3 must be normalized.");
#endif
	real_t d = p_axis.length();
	if (d == 0) {
		x = 0;
		y = 0;
		z = 0;
		w = 0;
	} else {
		real_t sin_angle = Math::sin(p_angle * 0.5);
		real_t cos_angle = Math::cos(p_angle * 0.5);
		real_t s = sin_angle / d;
		x = p_axis.x * s;
		y = p_axis.y * s;
		z = p_axis.z * s;
		w = cos_angle;
	}
}

// Euler constructor expects a vector containing the Euler angles in the format
// (ax, ay, az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
Quaternion::Quaternion(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.y * 0.5;
	real_t half_a2 = p_euler.x * 0.5;
	real_t half_a3 = p_euler.z * 0.5;

	// R = Y(a1).X(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = Math::cos(half_a1);
	real_t sin_a1 = Math::sin(half_a1);
	real_t cos_a2 = Math::cos(half_a2);
	real_t sin_a2 = Math::sin(half_a2);
	real_t cos_a3 = Math::cos(half_a3);
	real_t sin_a3 = Math::sin(half_a3);

	x = sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3;
	y = sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3;
	z = -sin_a1 * sin_a2 * cos_a3 + cos_a1 * cos_a2 * sin_a3;
	w = sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3;
}

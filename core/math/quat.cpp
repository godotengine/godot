/*************************************************************************/
/*  quat.cpp                                                             */
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
#include "quat.h"
#include "matrix3.h"
#include "print_string.h"

// set_euler expects a vector containing the Euler angles in the format
// (c,b,a), where a is the angle of the first rotation, and c is the last.
// The current implementation uses XYZ convention (Z is the first rotation).
void Quat::set_euler(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.x * 0.5;
	real_t half_a2 = p_euler.y * 0.5;
	real_t half_a3 = p_euler.z * 0.5;

	// R = X(a1).Y(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-2)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = Math::cos(half_a1);
	real_t sin_a1 = Math::sin(half_a1);
	real_t cos_a2 = Math::cos(half_a2);
	real_t sin_a2 = Math::sin(half_a2);
	real_t cos_a3 = Math::cos(half_a3);
	real_t sin_a3 = Math::sin(half_a3);

	set(sin_a1 * cos_a2 * cos_a3 + sin_a2 * sin_a3 * cos_a1,
			-sin_a1 * sin_a3 * cos_a2 + sin_a2 * cos_a1 * cos_a3,
			sin_a1 * sin_a2 * cos_a3 + sin_a3 * cos_a1 * cos_a2,
			-sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

// get_euler returns a vector containing the Euler angles in the format
// (a1,a2,a3), where a3 is the angle of the first rotation, and a1 is the last.
// The current implementation uses XYZ convention (Z is the first rotation).
Vector3 Quat::get_euler() const {
	Basis m(*this);
	return m.get_euler();
}

void Quat::operator*=(const Quat &q) {

	set(w * q.x + x * q.w + y * q.z - z * q.y,
			w * q.y + y * q.w + z * q.x - x * q.z,
			w * q.z + z * q.w + x * q.y - y * q.x,
			w * q.w - x * q.x - y * q.y - z * q.z);
}

Quat Quat::operator*(const Quat &q) const {

	Quat r = *this;
	r *= q;
	return r;
}

real_t Quat::length() const {

	return Math::sqrt(length_squared());
}

void Quat::normalize() {
	*this /= length();
}

Quat Quat::normalized() const {
	return *this / length();
}

Quat Quat::inverse() const {
	return Quat(-x, -y, -z, w);
}

Quat Quat::slerp(const Quat &q, const real_t &t) const {

#if 0


	Quat dst=q;
	Quat src=*this;

	src.normalize();
	dst.normalize();

	real_t cosine = dst.dot(src);

	if (cosine < 0 && true) {
		cosine = -cosine;
		dst = -dst;
	} else {
		dst = dst;
	}

	if (Math::abs(cosine) < 1 - CMP_EPSILON) {
		// Standard case (slerp)
		real_t sine = Math::sqrt(1 - cosine*cosine);
		real_t angle = Math::atan2(sine, cosine);
		real_t inv_sine = 1.0 / sine;
		real_t coeff_0 = Math::sin((1.0 - t) * angle) * inv_sine;
		real_t coeff_1 = Math::sin(t * angle) * inv_sine;
		Quat ret=  src * coeff_0 + dst * coeff_1;

		return ret;
	} else {
		// There are two situations:
		// 1. "rkP" and "q" are very close (cosine ~= +1), so we can do a linear
		//    interpolation safely.
		// 2. "rkP" and "q" are almost invedste of each other (cosine ~= -1), there
		//    are an infinite number of possibilities interpolation. but we haven't
		//    have method to fix this case, so just use linear interpolation here.
		Quat ret =  src * (1.0 - t) + dst *t;
		// taking the complement requires renormalisation
		ret.normalize();
		return ret;
	}
#else

	Quat to1;
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = dot(q);

	// adjust signs (if necessary)
	if (cosom < 0.0) {
		cosom = -cosom;
		to1.x = -q.x;
		to1.y = -q.y;
		to1.z = -q.z;
		to1.w = -q.w;
	} else {
		to1.x = q.x;
		to1.y = q.y;
		to1.z = q.z;
		to1.w = q.w;
	}

	// calculate coefficients

	if ((1.0 - cosom) > CMP_EPSILON) {
		// standard case (slerp)
		omega = Math::acos(cosom);
		sinom = Math::sin(omega);
		scale0 = Math::sin((1.0 - t) * omega) / sinom;
		scale1 = Math::sin(t * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1.0 - t;
		scale1 = t;
	}
	// calculate final values
	return Quat(
			scale0 * x + scale1 * to1.x,
			scale0 * y + scale1 * to1.y,
			scale0 * z + scale1 * to1.z,
			scale0 * w + scale1 * to1.w);
#endif
}

Quat Quat::slerpni(const Quat &q, const real_t &t) const {

	const Quat &from = *this;

	real_t dot = from.dot(q);

	if (Math::absf(dot) > 0.9999) return from;

	real_t theta = Math::acos(dot),
		   sinT = 1.0 / Math::sin(theta),
		   newFactor = Math::sin(t * theta) * sinT,
		   invFactor = Math::sin((1.0 - t) * theta) * sinT;

	return Quat(invFactor * from.x + newFactor * q.x,
			invFactor * from.y + newFactor * q.y,
			invFactor * from.z + newFactor * q.z,
			invFactor * from.w + newFactor * q.w);

#if 0
	real_t         to1[4];
	real_t        omega, cosom, sinom, scale0, scale1;


	// calc cosine
	cosom = x * q.x + y * q.y + z * q.z
			+ w * q.w;


	// adjust signs (if necessary)
	if ( cosom <0.0 && false) {
		cosom = -cosom;to1[0] = - q.x;
		to1[1] = - q.y;
		to1[2] = - q.z;
		to1[3] = - q.w;
	} else  {
		to1[0] = q.x;
		to1[1] = q.y;
		to1[2] = q.z;
		to1[3] = q.w;
	}


	// calculate coefficients

	if ( (1.0 - cosom) > CMP_EPSILON ) {
		// standard case (slerp)
		omega = Math::acos(cosom);
		sinom = Math::sin(omega);
		scale0 = Math::sin((1.0 - t) * omega) / sinom;
		scale1 = Math::sin(t * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1.0 - t;
		scale1 = t;
	}
	// calculate final values
	return Quat(
		scale0 * x + scale1 * to1[0],
		scale0 * y + scale1 * to1[1],
		scale0 * z + scale1 * to1[2],
		scale0 * w + scale1 * to1[3]
	);
#endif
}

Quat Quat::cubic_slerp(const Quat &q, const Quat &prep, const Quat &postq, const real_t &t) const {

	//the only way to do slerp :|
	real_t t2 = (1.0 - t) * t * 2;
	Quat sp = this->slerp(q, t);
	Quat sq = prep.slerpni(postq, t);
	return sp.slerpni(sq, t2);
}

Quat::operator String() const {

	return String::num(x) + ", " + String::num(y) + ", " + String::num(z) + ", " + String::num(w);
}

Quat::Quat(const Vector3 &axis, const real_t &angle) {
	real_t d = axis.length();
	if (d == 0)
		set(0, 0, 0, 0);
	else {
		real_t sin_angle = Math::sin(angle * 0.5);
		real_t cos_angle = Math::cos(angle * 0.5);
		real_t s = sin_angle / d;
		set(axis.x * s, axis.y * s, axis.z * s,
				cos_angle);
	}
}

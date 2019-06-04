/*************************************************************************/
/*  quat.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "print_string.h"

void Quat::set_euler(const Vector3 &p_euler) {
	real_t half_yaw = p_euler.x * 0.5;
	real_t half_pitch = p_euler.y * 0.5;
	real_t half_roll = p_euler.z * 0.5;
	real_t cos_yaw = Math::cos(half_yaw);
	real_t sin_yaw = Math::sin(half_yaw);
	real_t cos_pitch = Math::cos(half_pitch);
	real_t sin_pitch = Math::sin(half_pitch);
	real_t cos_roll = Math::cos(half_roll);
	real_t sin_roll = Math::sin(half_roll);
	set(cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
			cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
			sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
			cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw);
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
		real_t inv_sine = 1.0f / sine;
		real_t coeff_0 = Math::sin((1.0f - t) * angle) * inv_sine;
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
		Quat ret =  src * (1.0f - t) + dst *t;
		// taking the complement requires renormalisation
		ret.normalize();
		return ret;
	}
#else

	real_t to1[4];
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = x * q.x + y * q.y + z * q.z + w * q.w;

	// adjust signs (if necessary)
	if (cosom < 0.0) {
		cosom = -cosom;
		to1[0] = -q.x;
		to1[1] = -q.y;
		to1[2] = -q.z;
		to1[3] = -q.w;
	} else {
		to1[0] = q.x;
		to1[1] = q.y;
		to1[2] = q.z;
		to1[3] = q.w;
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
			scale0 * x + scale1 * to1[0],
			scale0 * y + scale1 * to1[1],
			scale0 * z + scale1 * to1[2],
			scale0 * w + scale1 * to1[3]);
#endif
}

Quat Quat::slerpni(const Quat &q, const real_t &t) const {

	const Quat &from = *this;

	float dot = from.dot(q);

	if (Math::absf(dot) > 0.9999f) return from;

	float theta = Math::acos(dot),
		  sinT = 1.0f / Math::sin(theta),
		  newFactor = Math::sin(t * theta) * sinT,
		  invFactor = Math::sin((1.0f - t) * theta) * sinT;

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
		cosom = -cosom; to1[0] = - q.x;
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
	float t2 = (1.0 - t) * t * 2;
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
		real_t s = Math::sin(-angle * 0.5) / d;
		set(axis.x * s, axis.y * s, axis.z * s,
				Math::cos(-angle * 0.5));
	}
}

/**************************************************************************/
/*  vector2.cpp                                                           */
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

#include "vector2.h"

#include "core/math/vector2i.h"
#include "core/string/ustring.h"

real_t Vector2::angle() const {
	return Math::atan2(y, x);
}

Vector2 Vector2::from_angle(real_t p_angle) {
	return Vector2(Math::cos(p_angle), Math::sin(p_angle));
}

bool Vector2::is_normalized() const {
	// Use length_squared() instead of length() to avoid sqrt(), makes it more stringent.
	return Math::is_equal_approx(length_squared(), 1, (real_t)UNIT_EPSILON);
}

real_t Vector2::angle_to(const Vector2 &p_vector2) const {
	return Math::atan2(cross(p_vector2), dot(p_vector2));
}

real_t Vector2::angle_to_point(const Vector2 &p_vector2) const {
	return (p_vector2 - *this).angle();
}

Vector2 Vector2::rotated(real_t p_by) const {
	real_t sine = Math::sin(p_by);
	real_t cosi = Math::cos(p_by);
	return Vector2(
			x * cosi - y * sine,
			x * sine + y * cosi);
}

Vector2 Vector2::posmod(real_t p_mod) const {
	return Vector2(Math::fposmod(x, p_mod), Math::fposmod(y, p_mod));
}

Vector2 Vector2::posmodv(const Vector2 &p_modv) const {
	return Vector2(Math::fposmod(x, p_modv.x), Math::fposmod(y, p_modv.y));
}

Vector2 Vector2::project(const Vector2 &p_to) const {
	return p_to * (dot(p_to) / p_to.length_squared());
}

Vector2 Vector2::snapped(const Vector2 &p_step) const {
	return Vector2(
			Math::snapped(x, p_step.x),
			Math::snapped(y, p_step.y));
}

Vector2 Vector2::snappedf(real_t p_step) const {
	return Vector2(
			Math::snapped(x, p_step),
			Math::snapped(y, p_step));
}

Vector2 Vector2::limit_length(real_t p_len) const {
	const real_t l = length();
	Vector2 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}
	return v;
}

Vector2 Vector2::move_toward(const Vector2 &p_to, real_t p_delta) const {
	Vector2 v = *this;
	Vector2 vd = p_to - v;
	real_t len = vd.length();
	return len <= p_delta || len < (real_t)CMP_EPSILON ? p_to : v + vd / len * p_delta;
}

// slide returns the component of the vector along the given plane, specified by its normal vector.
Vector2 Vector2::slide(const Vector2 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector2(), "The normal Vector2 " + p_normal.operator String() + "must be normalized.");
#endif
	return *this - p_normal * dot(p_normal);
}

Vector2 Vector2::bounce(const Vector2 &p_normal) const {
	return -reflect(p_normal);
}

Vector2 Vector2::reflect(const Vector2 &p_normal) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_normal.is_normalized(), Vector2(), "The normal Vector2 " + p_normal.operator String() + "must be normalized.");
#endif
	return 2.0f * p_normal * dot(p_normal) - *this;
}

Vector2 Vector2::slerp(const Vector2 &p_to, real_t p_weight) const {
	real_t start_length_sq = length_squared();
	real_t end_length_sq = p_to.length_squared();
	if (unlikely(start_length_sq == 0.0f || end_length_sq == 0.0f)) {
		// Zero length vectors have no angle, so the best we can do is either lerp or throw an error.
		return lerp(p_to, p_weight);
	}
	real_t start_length = Math::sqrt(start_length_sq);
	real_t result_length = Math::lerp(start_length, Math::sqrt(end_length_sq), p_weight);
	real_t angle = angle_to(p_to);
	return rotated(angle * p_weight) * (result_length / start_length);
}

Vector2::operator String() const {
	return "(" + String::num_real(x, false) + ", " + String::num_real(y, false) + ")";
}

Vector2::operator Vector2i() const {
	return Vector2i(x, y);
}

Vector2 Vector2::cubic_interpolate(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, real_t p_weight) const {
	return Vector2(
			Math::cubic_interpolate(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight),
			Math::cubic_interpolate(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight));
}

Vector2 Vector2::cubic_interpolate_in_time(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, real_t p_weight, real_t p_b_t, real_t p_pre_a_t, real_t p_post_b_t) const {
	return Vector2(
			Math::cubic_interpolate_in_time(x, p_b.x, p_pre_a.x, p_post_b.x, p_weight, p_b_t, p_pre_a_t, p_post_b_t),
			Math::cubic_interpolate_in_time(y, p_b.y, p_pre_a.y, p_post_b.y, p_weight, p_b_t, p_pre_a_t, p_post_b_t));
}

Vector2 Vector2::bezier_interpolate(const Vector2 &p_control_1, const Vector2 &p_control_2, const Vector2 &p_end, real_t p_t) const {
	return Vector2(
			Math::bezier_interpolate(x, p_control_1.x, p_control_2.x, p_end.x, p_t),
			Math::bezier_interpolate(y, p_control_1.y, p_control_2.y, p_end.y, p_t));
}

Vector2 Vector2::bezier_derivative(const Vector2 &p_control_1, const Vector2 &p_control_2, const Vector2 &p_end, real_t p_t) const {
	return Vector2(
			Math::bezier_derivative(x, p_control_1.x, p_control_2.x, p_end.x, p_t),
			Math::bezier_derivative(y, p_control_1.y, p_control_2.y, p_end.y, p_t));
}

Vector2 Vector2::plane_project(real_t p_d, const Vector2 &p_vec) const {
	return p_vec - *this * (dot(p_vec) - p_d);
}

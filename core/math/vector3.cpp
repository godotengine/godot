/*************************************************************************/
/*  vector3.cpp                                                          */
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

#include "vector3.h"

#include "core/math/basis.h"

void Vector3::rotate(const Vector3 &p_axis, const real_t p_phi) {
	*this = Basis(p_axis, p_phi).xform(*this);
}

Vector3 Vector3::rotated(const Vector3 &p_axis, const real_t p_phi) const {
	Vector3 r = *this;
	r.rotate(p_axis, p_phi);
	return r;
}

void Vector3::set_axis(const int p_axis, const real_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}

real_t Vector3::get_axis(const int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

Vector3 Vector3::clamp(const Vector3 &p_min, const Vector3 &p_max) const {
	return Vector3(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z));
}

void Vector3::snap(const Vector3 p_step) {
	x = Math::snapped(x, p_step.x);
	y = Math::snapped(y, p_step.y);
	z = Math::snapped(z, p_step.z);
}

Vector3 Vector3::snapped(const Vector3 p_step) const {
	Vector3 v = *this;
	v.snap(p_step);
	return v;
}

Vector3 Vector3::limit_length(const real_t p_len) const {
	const real_t l = length();
	Vector3 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}

	return v;
}

Vector3 Vector3::cubic_interpolate(const Vector3 &p_b, const Vector3 &p_pre_a, const Vector3 &p_post_b, const real_t p_weight) const {
	Vector3 p0 = p_pre_a;
	Vector3 p1 = *this;
	Vector3 p2 = p_b;
	Vector3 p3 = p_post_b;

	real_t t = p_weight;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	Vector3 out;
	out = 0.5 *
			((p1 * 2.0) +
					(-p0 + p2) * t +
					(2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
					(-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
	return out;
}

Vector3 Vector3::move_toward(const Vector3 &p_to, const real_t p_delta) const {
	Vector3 v = *this;
	Vector3 vd = p_to - v;
	real_t len = vd.length();
	return len <= p_delta || len < CMP_EPSILON ? p_to : v + vd / len * p_delta;
}

Basis Vector3::outer(const Vector3 &p_b) const {
	Vector3 row0(x * p_b.x, x * p_b.y, x * p_b.z);
	Vector3 row1(y * p_b.x, y * p_b.y, y * p_b.z);
	Vector3 row2(z * p_b.x, z * p_b.y, z * p_b.z);

	return Basis(row0, row1, row2);
}

bool Vector3::is_equal_approx(const Vector3 &p_v) const {
	return Math::is_equal_approx(x, p_v.x) && Math::is_equal_approx(y, p_v.y) && Math::is_equal_approx(z, p_v.z);
}

Vector3::operator String() const {
	return "(" + String::num_real(x, false) + ", " + String::num_real(y, false) + ", " + String::num_real(z, false) + ")";
}

Vector3 Vector3::get_xyz() const {
	return *this;
}

Vector3 Vector3::get_xzy() const {
	return Vector3(x, z, y);
}

Vector3 Vector3::get_yxz() const {
	return Vector3(y, x, z);
}

Vector3 Vector3::get_yzx() const {
	return Vector3(y, z, x);
}

Vector3 Vector3::get_zxy() const {
	return Vector3(z, x, y);
}

Vector3 Vector3::get_zyx() const {
	return Vector3(z, y, x);
}

Vector2 Vector3::get_xy() const {
	return Vector2(x, y);
}

Vector2 Vector3::get_xz() const {
	return Vector2(x, z);
}

Vector2 Vector3::get_yx() const {
	return Vector2(y, x);
}

Vector2 Vector3::get_yz() const {
	return Vector2(y, z);
}

Vector2 Vector3::get_zx() const {
	return Vector2(z, x);
}

Vector2 Vector3::get_zy() const {
	return Vector2(z, y);
}

/**
void Vector3::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_xyz"), &Vector3::get_xyz);
	ClassDB::bind_method(D_METHOD("get_xzy"), &Vector3::get_xzy);
	ClassDB::bind_method(D_METHOD("get_yxz"), &Vector3::get_yxz);
	ClassDB::bind_method(D_METHOD("get_yzx"), &Vector3::get_yzx);
	ClassDB::bind_method(D_METHOD("get_zxy"), &Vector3::get_zxy);
	ClassDB::bind_method(D_METHOD("get_zyx"), &Vector3::get_zyx);

	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "xyz"), "get_xyz");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "xzy"), "get_xzy");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "yxz"), "get_yxz");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "yzx"), "get_yzx");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "zxy"), "get_zxy");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "zyx"), "get_zyx");

	ClassDB::bind_method(D_METHOD("get_xy"), &Vector3::get_xy);
	ClassDB::bind_method(D_METHOD("get_xz"), &Vector3::get_xz);
	ClassDB::bind_method(D_METHOD("get_yx"), &Vector3::get_yx);
	ClassDB::bind_method(D_METHOD("get_yz"), &Vector3::get_yz);
	ClassDB::bind_method(D_METHOD("get_zx"), &Vector3::get_zx);
	ClassDB::bind_method(D_METHOD("get_zy"), &Vector3::get_zy);

	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "xy"), "get_xy");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "xz"), "get_xz");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "yx"), "get_yx");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "yz"), "get_yz");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "zx"), "get_zx");
	ADD_PROPERTY(PropertyInfo(Variant::Vector3, "zy"), "get_zy");
}
**/

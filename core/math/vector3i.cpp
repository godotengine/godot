/*************************************************************************/
/*  vector3i.cpp                                                         */
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

#include "vector3i.h"
#include "core/math/vector2.h"

void Vector3i::set_axis(const int p_axis, const int32_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}

int32_t Vector3i::get_axis(const int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

int Vector3i::min_axis() const {
	return x < y ? (x < z ? 0 : 2) : (y < z ? 1 : 2);
}

int Vector3i::max_axis() const {
	return x < y ? (y < z ? 2 : 1) : (x < z ? 2 : 0);
}

Vector3i Vector3i::clamp(const Vector3i &p_min, const Vector3i &p_max) const {
	return Vector3i(
			CLAMP(x, p_min.x, p_max.x),
			CLAMP(y, p_min.y, p_max.y),
			CLAMP(z, p_min.z, p_max.z));
}

Vector3i::operator String() const {
	return "(" + itos(x) + ", " + itos(y) + ", " + itos(z) + ")";
}

Vector3i Vector3i::get_xyz() const {
	return *this;
}

Vector3i Vector3i::get_xzy() const {
	return Vector3i(x, z, y);
}

Vector3i Vector3i::get_yxz() const {
	return Vector3i(y, x, z);
}

Vector3i Vector3i::get_yzx() const {
	return Vector3i(y, z, x);
}

Vector3i Vector3i::get_zxy() const {
	return Vector3i(z, x, y);
}

Vector3i Vector3i::get_zyx() const {
	return Vector3i(z, y, x);
}

Vector2i Vector3i::get_xy() const {
	return Vector2i(x, y);
}

Vector2i Vector3i::get_xz() const {
	return Vector2i(x, z);
}

Vector2i Vector3i::get_yx() const {
	return Vector2i(y, x);
}

Vector2i Vector3i::get_yz() const {
	return Vector2i(y, z);
}

Vector2i Vector3i::get_zx() const {
	return Vector2i(z, x);
}

Vector2i Vector3i::get_zy() const {
	return Vector2i(z, y);
}

void Vector3i::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_xyz"), &Vector3i::get_xyz);
	ClassDB::bind_method(D_METHOD("get_xzy"), &Vector3i::get_xzy);
	ClassDB::bind_method(D_METHOD("get_yxz"), &Vector3i::get_yxz);
	ClassDB::bind_method(D_METHOD("get_yzx"), &Vector3i::get_yzx);
	ClassDB::bind_method(D_METHOD("get_zxy"), &Vector3i::get_zxy);
	ClassDB::bind_method(D_METHOD("get_zyx"), &Vector3i::get_zyx);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "xyz"), "get_xyz");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "xzy"), "get_xzy");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "yxz"), "get_yxz");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "yzx"), "get_yzx");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "zxy"), "get_zxy");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "zyx"), "get_zyx");

	ClassDB::bind_method(D_METHOD("get_xy"), &Vector3i::get_xy);
	ClassDB::bind_method(D_METHOD("get_xz"), &Vector3i::get_xz);
	ClassDB::bind_method(D_METHOD("get_yx"), &Vector3i::get_yx);
	ClassDB::bind_method(D_METHOD("get_yz"), &Vector3i::get_yz);
	ClassDB::bind_method(D_METHOD("get_zx"), &Vector3i::get_zx);
	ClassDB::bind_method(D_METHOD("get_zy"), &Vector3i::get_zy);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "xy"), "get_xy");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "xz"), "get_xz");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "yx"), "get_yx");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "yz"), "get_yz");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "zx"), "get_zx");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "zy"), "get_zy");
}

/*************************************************************************/
/*  vector3i.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

void Vector3i::set_axis(int p_axis, int32_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	coord[p_axis] = p_value;
}

int32_t Vector3i::get_axis(int p_axis) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	return operator[](p_axis);
}

int Vector3i::min_axis() const {
	return x < y ? (x < z ? 0 : 2) : (y < z ? 1 : 2);
}

int Vector3i::max_axis() const {
	return x < y ? (y < z ? 2 : 1) : (x < z ? 2 : 0);
}

Vector3i Vector3i::cross(const Vector3i &p_b) const {

	Vector3i ret(
			(y * p_b.z) - (z * p_b.y),
			(z * p_b.x) - (x * p_b.z),
			(x * p_b.y) - (y * p_b.x));

	return ret;
}

int64_t Vector3i::dot(const Vector3i &p_b) const {

	return x * p_b.x + y * p_b.y + z * p_b.z;
}

real_t Vector3i::length() const {

	int64_t x2 = x * x;
	int64_t y2 = y * y;
	int64_t z2 = z * z;

	return Math::sqrt((real_t)(x2 + y2 + z2));
}

int64_t Vector3i::length_squared() const {

	int64_t x2 = x * x;
	int64_t y2 = y * y;
	int64_t z2 = z * z;

	return x2 + y2 + z2;
}

real_t Vector3i::distance_to(const Vector3i &p_b) const {

	return (p_b - *this).length();
}

int64_t Vector3i::distance_squared_to(const Vector3i &p_b) const {

	return (p_b - *this).length_squared();
}

Vector3i Vector3i::posmod(const int32_t p_mod) const {
	return Vector3i(Math::posmod(x, p_mod), Math::posmod(y, p_mod), Math::posmod(z, p_mod));
}

Vector3i Vector3i::posmodv(const Vector3i &p_modv) const {
	return Vector3i(Math::posmod(x, p_modv.x), Math::posmod(y, p_modv.y), Math::posmod(z, p_modv.z));
}

Vector3i::operator String() const {
	return (itos(x) + ", " + itos(y) + ", " + itos(z));
}

Vector3i::operator Vector3() const {
	return Vector3(x, y, z);
}

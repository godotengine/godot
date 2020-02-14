/*************************************************************************/
/*  vector2i.cpp                                                         */
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

#include "vector2.h"

int Vector2i::min_axis() const {
	return x < y ? AXIS_X : AXIS_Y;
}

int Vector2i::max_axis() const {
	return x < y ? AXIS_Y : AXIS_X;
}

real_t Vector2i::length() const {
	int64_t x2 = x * x;
	int64_t y2 = y * y;
	return Math::sqrt((real_t)(x2 + y2));
}

int64_t Vector2i::length_squared() const {
	int64_t x2 = x * x;
	int64_t y2 = y * y;
	return x2 + y2;
}

real_t Vector2i::distance_to(const Vector2i &p_b) const {
	return (p_b - *this).length();
}

int64_t Vector2i::distance_squared_to(const Vector2i &p_b) const {
	return (p_b - *this).length_squared();
}

Vector2i Vector2i::abs() const {
	return Vector2i(Math::abs(x), Math::abs(y));
}

Vector2i Vector2i::sign() const {
	return Vector2i(SGN(x), SGN(y));
}

int64_t Vector2i::dot(const Vector2i &p_b) const {
	return x * p_b.x + y * p_b.y;
}

int64_t Vector2i::cross(const Vector2i &p_other) const {
	return x * p_other.y - y * p_other.x;
}

Vector2i Vector2i::posmod(const real_t p_mod) const {
	return Vector2i(Math::posmod(x, p_mod), Math::posmod(y, p_mod));
}

Vector2i Vector2i::posmodv(const Vector2i &p_modv) const {
	return Vector2i(Math::posmod(x, p_modv.x), Math::posmod(y, p_modv.y));
}

Vector2i Vector2i::operator+(const Vector2i &p_v) const {
	return Vector2i(x + p_v.x, y + p_v.y);
}

void Vector2i::operator+=(const Vector2i &p_v) {
	x += p_v.x;
	y += p_v.y;
}

Vector2i Vector2i::operator-(const Vector2i &p_v) const {
	return Vector2i(x - p_v.x, y - p_v.y);
}

void Vector2i::operator-=(const Vector2i &p_v) {
	x -= p_v.x;
	y -= p_v.y;
}

Vector2i Vector2i::operator*(const Vector2i &p_v1) const {
	return Vector2i(x * p_v1.x, y * p_v1.y);
};

Vector2i Vector2i::operator*(const int32_t &rvalue) const {
	return Vector2i(x * rvalue, y * rvalue);
};

void Vector2i::operator*=(const int32_t &rvalue) {
	x *= rvalue;
	y *= rvalue;
};

Vector2i Vector2i::operator/(const Vector2i &p_v1) const {
	return Vector2i(x / p_v1.x, y / p_v1.y);
};

Vector2i Vector2i::operator/(const int32_t &rvalue) const {
	return Vector2i(x / rvalue, y / rvalue);
};

void Vector2i::operator/=(const int32_t &rvalue) {
	x /= rvalue;
	y /= rvalue;
};

Vector2i Vector2i::operator-() const {
	return Vector2i(-x, -y);
}

bool Vector2i::operator==(const Vector2i &p_vec2i) const {
	return x == p_vec2i.x && y == p_vec2i.y;
}

bool Vector2i::operator!=(const Vector2i &p_vec2i) const {
	return x != p_vec2i.x || y != p_vec2i.y;
}

Vector2i::operator Vector2() const {
	return Vector2(x, y);
}

/*************************************************************************/
/*  vector2i.h                                                           */
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

#ifndef VECTOR2I_H
#define VECTOR2I_H

#include "core/math/math_funcs.h"
#include "core/ustring.h"

struct Vector2;

struct Vector2i {

	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	union {
		int32_t x;
		int32_t width;
	};
	union {
		int32_t y;
		int32_t height;
	};

	_FORCE_INLINE_ int32_t &operator[](int p_idx) {
		return p_idx ? y : x;
	}
	_FORCE_INLINE_ const int32_t &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	int min_axis() const;
	int max_axis() const;

	Vector2i abs() const;
	Vector2i sign() const;

	// Outputs can potentially be much bigger, so use int64_t instead of int32_t.
	int64_t cross(const Vector2i &p_other) const;
	int64_t dot(const Vector2i &p_b) const;
	real_t length() const;
	int64_t length_squared() const;
	real_t distance_to(const Vector2i &p_b) const;
	int64_t distance_squared_to(const Vector2i &p_b) const;
	Vector2i tangent() const {
		return Vector2i(y, -x);
	}
	Vector2i posmod(const real_t p_mod) const;
	Vector2i posmodv(const Vector2i &p_modv) const;

	Vector2i operator+(const Vector2i &p_v) const;
	void operator+=(const Vector2i &p_v);
	Vector2i operator-(const Vector2i &p_v) const;
	void operator-=(const Vector2i &p_v);
	Vector2i operator*(const Vector2i &p_v1) const;

	Vector2i operator*(const int32_t &rvalue) const;
	void operator*=(const int32_t &rvalue);

	Vector2i operator/(const Vector2i &p_v1) const;

	Vector2i operator/(const int32_t &rvalue) const;

	void operator/=(const int32_t &rvalue);

	Vector2i operator-() const;
	bool operator<(const Vector2i &p_vec2i) const { return (x == p_vec2i.x) ? (y < p_vec2i.y) : (x < p_vec2i.x); }
	bool operator>(const Vector2i &p_vec2i) const { return (x == p_vec2i.x) ? (y > p_vec2i.y) : (x > p_vec2i.x); }
	bool operator<=(const Vector2i &p_vec2i) const { return (x == p_vec2i.x) ? (y <= p_vec2i.y) : (x < p_vec2i.x); }
	bool operator>=(const Vector2i &p_vec2i) const { return (x == p_vec2i.x) ? (y >= p_vec2i.y) : (x > p_vec2i.x); }

	bool operator==(const Vector2i &p_vec2i) const;
	bool operator!=(const Vector2i &p_vec2i) const;

	real_t aspect() const { return width / (real_t)height; }

	operator String() const { return String::num(x) + ", " + String::num(y); }
	operator Vector2() const;

	inline Vector2i(int32_t p_x, int32_t p_y) {
		x = p_x;
		y = p_y;
	}
	inline Vector2i() { x = y = 0; }
};

typedef Vector2i Point2i;
typedef Vector2i Size2i;

#endif // VECTOR2I_H

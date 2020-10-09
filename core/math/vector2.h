/*************************************************************************/
/*  vector2.h                                                            */
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

#ifndef VECTOR2_H
#define VECTOR2_H

#include "core/math/math_funcs.h"
#include "core/ustring.h"

struct Vector2i;

struct Vector2 {
	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	union {
		real_t x = 0;
		real_t width;
	};
	union {
		real_t y = 0;
		real_t height;
	};

	_FORCE_INLINE_ real_t &operator[](int p_idx) {
		return p_idx ? y : x;
	}
	_FORCE_INLINE_ const real_t &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	void normalize();
	Vector2 normalized() const;
	bool is_normalized() const;

	real_t length() const;
	real_t length_squared() const;

	Vector2 min(const Vector2 &p_vector2) const {
		return Vector2(MIN(x, p_vector2.x), MIN(y, p_vector2.y));
	}

	Vector2 max(const Vector2 &p_vector2) const {
		return Vector2(MAX(x, p_vector2.x), MAX(y, p_vector2.y));
	}

	real_t distance_to(const Vector2 &p_vector2) const;
	real_t distance_squared_to(const Vector2 &p_vector2) const;
	real_t angle_to(const Vector2 &p_vector2) const;
	real_t angle_to_point(const Vector2 &p_vector2) const;
	_FORCE_INLINE_ Vector2 direction_to(const Vector2 &p_b) const;

	real_t dot(const Vector2 &p_other) const;
	real_t cross(const Vector2 &p_other) const;
	Vector2 posmod(const real_t p_mod) const;
	Vector2 posmodv(const Vector2 &p_modv) const;
	Vector2 project(const Vector2 &p_b) const;

	Vector2 plane_project(real_t p_d, const Vector2 &p_vec) const;

	Vector2 clamped(real_t p_len) const;

	_FORCE_INLINE_ Vector2 lerp(const Vector2 &p_b, real_t p_t) const;
	_FORCE_INLINE_ Vector2 slerp(const Vector2 &p_b, real_t p_t) const;
	Vector2 cubic_interpolate(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, real_t p_t) const;
	Vector2 move_toward(const Vector2 &p_to, const real_t p_delta) const;

	Vector2 slide(const Vector2 &p_normal) const;
	Vector2 bounce(const Vector2 &p_normal) const;
	Vector2 reflect(const Vector2 &p_normal) const;

	bool is_equal_approx(const Vector2 &p_v) const;

	Vector2 operator+(const Vector2 &p_v) const;
	void operator+=(const Vector2 &p_v);
	Vector2 operator-(const Vector2 &p_v) const;
	void operator-=(const Vector2 &p_v);
	Vector2 operator*(const Vector2 &p_v1) const;

	Vector2 operator*(const real_t &rvalue) const;
	void operator*=(const real_t &rvalue);
	void operator*=(const Vector2 &rvalue) { *this = *this * rvalue; }

	Vector2 operator/(const Vector2 &p_v1) const;

	Vector2 operator/(const real_t &rvalue) const;

	void operator/=(const real_t &rvalue);
	void operator/=(const Vector2 &rvalue) { *this = *this / rvalue; }

	Vector2 operator-() const;

	bool operator==(const Vector2 &p_vec2) const;
	bool operator!=(const Vector2 &p_vec2) const;

	bool operator<(const Vector2 &p_vec2) const { return x == p_vec2.x ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator>(const Vector2 &p_vec2) const { return x == p_vec2.x ? (y > p_vec2.y) : (x > p_vec2.x); }
	bool operator<=(const Vector2 &p_vec2) const { return x == p_vec2.x ? (y <= p_vec2.y) : (x < p_vec2.x); }
	bool operator>=(const Vector2 &p_vec2) const { return x == p_vec2.x ? (y >= p_vec2.y) : (x > p_vec2.x); }

	real_t angle() const;

	_FORCE_INLINE_ Vector2 abs() const {
		return Vector2(Math::abs(x), Math::abs(y));
	}

	Vector2 rotated(real_t p_by) const;
	Vector2 tangent() const {
		return Vector2(y, -x);
	}

	Vector2 sign() const;
	Vector2 floor() const;
	Vector2 ceil() const;
	Vector2 round() const;
	Vector2 snapped(const Vector2 &p_by) const;
	real_t aspect() const { return width / height; }

	operator String() const { return String::num(x) + ", " + String::num(y); }

	_FORCE_INLINE_ Vector2() {}
	_FORCE_INLINE_ Vector2(real_t p_x, real_t p_y) {
		x = p_x;
		y = p_y;
	}
};

_FORCE_INLINE_ Vector2 Vector2::plane_project(real_t p_d, const Vector2 &p_vec) const {
	return p_vec - *this * (dot(p_vec) - p_d);
}

_FORCE_INLINE_ Vector2 operator*(float p_scalar, const Vector2 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector2 operator*(double p_scalar, const Vector2 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector2 operator*(int32_t p_scalar, const Vector2 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector2 operator*(int64_t p_scalar, const Vector2 &p_vec) {
	return p_vec * p_scalar;
}

_FORCE_INLINE_ Vector2 Vector2::operator+(const Vector2 &p_v) const {
	return Vector2(x + p_v.x, y + p_v.y);
}

_FORCE_INLINE_ void Vector2::operator+=(const Vector2 &p_v) {
	x += p_v.x;
	y += p_v.y;
}

_FORCE_INLINE_ Vector2 Vector2::operator-(const Vector2 &p_v) const {
	return Vector2(x - p_v.x, y - p_v.y);
}

_FORCE_INLINE_ void Vector2::operator-=(const Vector2 &p_v) {
	x -= p_v.x;
	y -= p_v.y;
}

_FORCE_INLINE_ Vector2 Vector2::operator*(const Vector2 &p_v1) const {
	return Vector2(x * p_v1.x, y * p_v1.y);
}

_FORCE_INLINE_ Vector2 Vector2::operator*(const real_t &rvalue) const {
	return Vector2(x * rvalue, y * rvalue);
}

_FORCE_INLINE_ void Vector2::operator*=(const real_t &rvalue) {
	x *= rvalue;
	y *= rvalue;
}

_FORCE_INLINE_ Vector2 Vector2::operator/(const Vector2 &p_v1) const {
	return Vector2(x / p_v1.x, y / p_v1.y);
}

_FORCE_INLINE_ Vector2 Vector2::operator/(const real_t &rvalue) const {
	return Vector2(x / rvalue, y / rvalue);
}

_FORCE_INLINE_ void Vector2::operator/=(const real_t &rvalue) {
	x /= rvalue;
	y /= rvalue;
}

_FORCE_INLINE_ Vector2 Vector2::operator-() const {
	return Vector2(-x, -y);
}

_FORCE_INLINE_ bool Vector2::operator==(const Vector2 &p_vec2) const {
	return x == p_vec2.x && y == p_vec2.y;
}

_FORCE_INLINE_ bool Vector2::operator!=(const Vector2 &p_vec2) const {
	return x != p_vec2.x || y != p_vec2.y;
}

Vector2 Vector2::lerp(const Vector2 &p_b, real_t p_t) const {
	Vector2 res = *this;

	res.x += (p_t * (p_b.x - x));
	res.y += (p_t * (p_b.y - y));

	return res;
}

Vector2 Vector2::slerp(const Vector2 &p_b, real_t p_t) const {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!is_normalized(), Vector2(), "The start Vector2 must be normalized.");
#endif
	real_t theta = angle_to(p_b);
	return rotated(theta * p_t);
}

Vector2 Vector2::direction_to(const Vector2 &p_b) const {
	Vector2 ret(p_b.x - x, p_b.y - y);
	ret.normalize();
	return ret;
}

typedef Vector2 Size2;
typedef Vector2 Point2;

/* INTEGER STUFF */

struct Vector2i {
	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	union {
		int x = 0;
		int width;
	};
	union {
		int y = 0;
		int height;
	};

	_FORCE_INLINE_ int &operator[](int p_idx) {
		return p_idx ? y : x;
	}
	_FORCE_INLINE_ const int &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	Vector2i operator+(const Vector2i &p_v) const;
	void operator+=(const Vector2i &p_v);
	Vector2i operator-(const Vector2i &p_v) const;
	void operator-=(const Vector2i &p_v);
	Vector2i operator*(const Vector2i &p_v1) const;

	Vector2i operator*(const int &rvalue) const;
	void operator*=(const int &rvalue);

	Vector2i operator/(const Vector2i &p_v1) const;

	Vector2i operator/(const int &rvalue) const;

	void operator/=(const int &rvalue);

	Vector2i operator-() const;
	bool operator<(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator>(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y > p_vec2.y) : (x > p_vec2.x); }

	bool operator<=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y <= p_vec2.y) : (x < p_vec2.x); }
	bool operator>=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y >= p_vec2.y) : (x > p_vec2.x); }

	bool operator==(const Vector2i &p_vec2) const;
	bool operator!=(const Vector2i &p_vec2) const;

	real_t aspect() const { return width / (real_t)height; }
	Vector2i sign() const { return Vector2i(SGN(x), SGN(y)); }
	Vector2i abs() const { return Vector2i(ABS(x), ABS(y)); }

	operator String() const { return String::num(x) + ", " + String::num(y); }

	operator Vector2() const { return Vector2(x, y); }

	inline Vector2i() {}
	inline Vector2i(const Vector2 &p_vec2) {
		x = (int)p_vec2.x;
		y = (int)p_vec2.y;
	}
	inline Vector2i(int p_x, int p_y) {
		x = p_x;
		y = p_y;
	}
};

_FORCE_INLINE_ Vector2i operator*(const int32_t &p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const int64_t &p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const float &p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

_FORCE_INLINE_ Vector2i operator*(const double &p_scalar, const Vector2i &p_vector) {
	return p_vector * p_scalar;
}

typedef Vector2i Size2i;
typedef Vector2i Point2i;

#endif // VECTOR2_H

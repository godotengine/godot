/**************************************************************************/
/*  vector4i.h                                                            */
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

#ifndef VECTOR4I_H
#define VECTOR4I_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"

class String;
struct Vector4;

struct _NO_DISCARD_ Vector4i {
	static constexpr int AXIS_COUNT = 4;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
		AXIS_W,
	};

	union {
		struct {
			int32_t x;
			int32_t y;
			int32_t z;
			int32_t w;
		};

		int32_t coord[4] = { 0, 0, 0, 0 };
	};

	constexpr const int32_t &operator[](size_t p_axis) const;
	constexpr int32_t &operator[](size_t p_axis);

	Vector4i::Axis min_axis_index() const;
	Vector4i::Axis max_axis_index() const;

	Vector4i min(const Vector4i &p_vector4i) const {
		return Vector4i(MIN(x, p_vector4i.x), MIN(y, p_vector4i.y), MIN(z, p_vector4i.z), MIN(w, p_vector4i.w));
	}

	Vector4i mini(int32_t p_scalar) const {
		return Vector4i(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar), MIN(w, p_scalar));
	}

	Vector4i max(const Vector4i &p_vector4i) const {
		return Vector4i(MAX(x, p_vector4i.x), MAX(y, p_vector4i.y), MAX(z, p_vector4i.z), MAX(w, p_vector4i.w));
	}

	Vector4i maxi(int32_t p_scalar) const {
		return Vector4i(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar), MAX(w, p_scalar));
	}

	_FORCE_INLINE_ int64_t length_squared() const;
	_FORCE_INLINE_ double length() const;

	_FORCE_INLINE_ void zero();

	_FORCE_INLINE_ double distance_to(const Vector4i &p_to) const;
	_FORCE_INLINE_ int64_t distance_squared_to(const Vector4i &p_to) const;

	_FORCE_INLINE_ Vector4i abs() const;
	_FORCE_INLINE_ Vector4i sign() const;
	Vector4i clamp(const Vector4i &p_min, const Vector4i &p_max) const;
	Vector4i clampi(int32_t p_min, int32_t p_max) const;
	Vector4i snapped(const Vector4i &p_step) const;
	Vector4i snappedi(int32_t p_step) const;

	constexpr Vector4i &operator+=(const Vector4i &p_other);
	constexpr Vector4i operator+(const Vector4i &p_other) const;
	constexpr Vector4i &operator-=(const Vector4i &p_other);
	constexpr Vector4i operator-(const Vector4i &p_other) const;
	constexpr Vector4i &operator*=(const Vector4i &p_other);
	constexpr Vector4i operator*(const Vector4i &p_other) const;
	constexpr Vector4i &operator/=(const Vector4i &p_other);
	constexpr Vector4i operator/(const Vector4i &p_other) const;
	constexpr Vector4i &operator%=(const Vector4i &p_other);
	constexpr Vector4i operator%(const Vector4i &p_other) const;

	constexpr Vector4i &operator*=(int32_t p_scalar);
	constexpr Vector4i operator*(int32_t p_scalar) const;
	constexpr Vector4i &operator/=(int32_t p_scalar);
	constexpr Vector4i operator/(int32_t p_scalar) const;
	constexpr Vector4i &operator%=(int32_t p_scalar);
	constexpr Vector4i operator%(int32_t p_scalar) const;

	constexpr Vector4i operator-() const;

	constexpr bool operator==(const Vector4i &p_other) const;
	constexpr bool operator!=(const Vector4i &p_other) const;
	constexpr bool operator<(const Vector4i &p_other) const;
	constexpr bool operator<=(const Vector4i &p_other) const;
	constexpr bool operator>(const Vector4i &p_other) const;
	constexpr bool operator>=(const Vector4i &p_other) const;

	operator String() const;
	operator Vector4() const;

	constexpr Vector4i() :
			x(0), y(0), z(0), w(0) {}

	constexpr Vector4i(int32_t p_x, int32_t p_y, int32_t p_z, int32_t p_w) :
			x(p_x), y(p_y), z(p_z), w(p_w) {}

	Vector4i(const Vector4 &p_vec4);
};

constexpr const int32_t &Vector4i::operator[](size_t p_axis) const {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_axis, AXIS_COUNT);
	}
#endif
	switch (p_axis) {
		case AXIS_X:
			return x;
		case AXIS_Y:
			return y;
		case AXIS_Z:
			return z;
		case AXIS_W:
			return w;
		default:
			return coord[p_axis];
	}
}

constexpr int32_t &Vector4i::operator[](size_t p_axis) {
#ifdef DEV_ENABLED
	if (!__builtin_is_constant_evaluated()) {
		CRASH_BAD_UNSIGNED_INDEX(p_axis, AXIS_COUNT);
	}
#endif
	switch (p_axis) {
		case AXIS_X:
			return x;
		case AXIS_Y:
			return y;
		case AXIS_Z:
			return z;
		case AXIS_W:
			return w;
		default:
			return coord[p_axis];
	}
}

constexpr Vector4i &Vector4i::operator+=(const Vector4i &p_other) {
	x += p_other.x;
	y += p_other.y;
	z += p_other.z;
	w += p_other.w;
	return *this;
}

constexpr Vector4i Vector4i::operator+(const Vector4i &p_other) const {
	return Vector4i(x + p_other.x, y + p_other.y, z + p_other.z, w + p_other.w);
}

constexpr Vector4i &Vector4i::operator-=(const Vector4i &p_other) {
	x -= p_other.x;
	y -= p_other.y;
	z -= p_other.z;
	w -= p_other.w;
	return *this;
}

constexpr Vector4i Vector4i::operator-(const Vector4i &p_other) const {
	return Vector4i(x - p_other.x, y - p_other.y, z - p_other.z, w - p_other.w);
}

constexpr Vector4i &Vector4i::operator*=(const Vector4i &p_other) {
	x *= p_other.x;
	y *= p_other.y;
	z *= p_other.z;
	w *= p_other.w;
	return *this;
}

constexpr Vector4i Vector4i::operator*(const Vector4i &p_other) const {
	return Vector4i(x * p_other.x, y * p_other.y, z * p_other.z, w * p_other.w);
}

constexpr Vector4i &Vector4i::operator/=(const Vector4i &p_other) {
	x /= p_other.x;
	y /= p_other.y;
	z /= p_other.z;
	w /= p_other.w;
	return *this;
}

constexpr Vector4i Vector4i::operator/(const Vector4i &p_other) const {
	return Vector4i(x / p_other.x, y / p_other.y, z / p_other.z, w / p_other.w);
}

constexpr Vector4i &Vector4i::operator%=(const Vector4i &p_other) {
	x %= p_other.x;
	y %= p_other.y;
	z %= p_other.z;
	w %= p_other.w;
	return *this;
}

constexpr Vector4i Vector4i::operator%(const Vector4i &p_other) const {
	return Vector4i(x % p_other.x, y % p_other.y, z % p_other.z, w % p_other.w);
}

constexpr Vector4i &Vector4i::operator*=(int32_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	w *= p_scalar;
	return *this;
}

constexpr Vector4i Vector4i::operator*(int32_t p_scalar) const {
	return Vector4i(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar);
}

constexpr Vector4i &Vector4i::operator/=(int32_t p_scalar) {
	x /= p_scalar;
	y /= p_scalar;
	z /= p_scalar;
	w /= p_scalar;
	return *this;
}

constexpr Vector4i Vector4i::operator/(int32_t p_scalar) const {
	return Vector4i(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar);
}

constexpr Vector4i &Vector4i::operator%=(int32_t p_scalar) {
	x %= p_scalar;
	y %= p_scalar;
	z %= p_scalar;
	w %= p_scalar;
	return *this;
}

constexpr Vector4i Vector4i::operator%(int32_t p_scalar) const {
	return Vector4i(x % p_scalar, y % p_scalar, z % p_scalar, w % p_scalar);
}

constexpr Vector4i Vector4i::operator-() const {
	return Vector4i(-x, -y, -z, -w);
}

constexpr bool Vector4i::operator==(const Vector4i &p_other) const {
	return (x == p_other.x && y == p_other.y && z == p_other.z && w == p_other.w);
}

constexpr bool Vector4i::operator!=(const Vector4i &p_other) const {
	return (x != p_other.x || y != p_other.y || z != p_other.z || w != p_other.w);
}

constexpr bool Vector4i::operator<(const Vector4i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w < p_other.w;
			}
			return z < p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector4i::operator>(const Vector4i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w > p_other.w;
			}
			return z > p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

constexpr bool Vector4i::operator<=(const Vector4i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w <= p_other.w;
			}
			return z < p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector4i::operator>=(const Vector4i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			if (z == p_other.z) {
				return w >= p_other.w;
			}
			return z > p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

// Multiplication operators required to workaround issues with LLVM using implicit conversion.

constexpr Vector4i operator*(int32_t p_scalar, const Vector4i &p_vector4i) {
	return p_vector4i * p_scalar;
}

constexpr Vector4i operator*(int64_t p_scalar, const Vector4i &p_vector4i) {
	return p_vector4i * p_scalar;
}

constexpr Vector4i operator*(float p_scalar, const Vector4i &p_vector4i) {
	return p_vector4i * p_scalar;
}

constexpr Vector4i operator*(double p_scalar, const Vector4i &p_vector4i) {
	return p_vector4i * p_scalar;
}

int64_t Vector4i::length_squared() const {
	return x * (int64_t)x + y * (int64_t)y + z * (int64_t)z + w * (int64_t)w;
}

double Vector4i::length() const {
	return Math::sqrt((double)length_squared());
}

double Vector4i::distance_to(const Vector4i &p_to) const {
	return (p_to - *this).length();
}

int64_t Vector4i::distance_squared_to(const Vector4i &p_to) const {
	return (p_to - *this).length_squared();
}

Vector4i Vector4i::abs() const {
	return Vector4i(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
}

Vector4i Vector4i::sign() const {
	return Vector4i(SIGN(x), SIGN(y), SIGN(z), SIGN(w));
}

void Vector4i::zero() {
	x = y = z = w = 0;
}

#endif // VECTOR4I_H

/**************************************************************************/
/*  vector3i.h                                                            */
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

#ifndef VECTOR3I_H
#define VECTOR3I_H

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"

class String;
struct Vector3;

struct _NO_DISCARD_ Vector3i {
	static constexpr int AXIS_COUNT = 3;

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	union {
		struct {
			int32_t x;
			int32_t y;
			int32_t z;
		};

		int32_t coord[3] = { 0, 0, 0 };
	};

	constexpr const int32_t &operator[](size_t p_axis) const;
	constexpr int32_t &operator[](size_t p_axis);

	Vector3i::Axis min_axis_index() const;
	Vector3i::Axis max_axis_index() const;

	Vector3i min(const Vector3i &p_vector3i) const {
		return Vector3i(MIN(x, p_vector3i.x), MIN(y, p_vector3i.y), MIN(z, p_vector3i.z));
	}

	Vector3i mini(int32_t p_scalar) const {
		return Vector3i(MIN(x, p_scalar), MIN(y, p_scalar), MIN(z, p_scalar));
	}

	Vector3i max(const Vector3i &p_vector3i) const {
		return Vector3i(MAX(x, p_vector3i.x), MAX(y, p_vector3i.y), MAX(z, p_vector3i.z));
	}

	Vector3i maxi(int32_t p_scalar) const {
		return Vector3i(MAX(x, p_scalar), MAX(y, p_scalar), MAX(z, p_scalar));
	}

	_FORCE_INLINE_ int64_t length_squared() const;
	_FORCE_INLINE_ double length() const;

	_FORCE_INLINE_ void zero();

	_FORCE_INLINE_ Vector3i abs() const;
	_FORCE_INLINE_ Vector3i sign() const;
	Vector3i clamp(const Vector3i &p_min, const Vector3i &p_max) const;
	Vector3i clampi(int32_t p_min, int32_t p_max) const;
	Vector3i snapped(const Vector3i &p_step) const;
	Vector3i snappedi(int32_t p_step) const;

	_FORCE_INLINE_ double distance_to(const Vector3i &p_to) const;
	_FORCE_INLINE_ int64_t distance_squared_to(const Vector3i &p_to) const;

	constexpr Vector3i &operator+=(const Vector3i &p_other);
	constexpr Vector3i operator+(const Vector3i &p_other) const;
	constexpr Vector3i &operator-=(const Vector3i &p_other);
	constexpr Vector3i operator-(const Vector3i &p_other) const;
	constexpr Vector3i &operator*=(const Vector3i &p_other);
	constexpr Vector3i operator*(const Vector3i &p_other) const;
	constexpr Vector3i &operator/=(const Vector3i &p_other);
	constexpr Vector3i operator/(const Vector3i &p_other) const;
	constexpr Vector3i &operator%=(const Vector3i &p_other);
	constexpr Vector3i operator%(const Vector3i &p_other) const;

	constexpr Vector3i &operator*=(int32_t p_scalar);
	constexpr Vector3i operator*(int32_t p_scalar) const;
	constexpr Vector3i &operator/=(int32_t p_scalar);
	constexpr Vector3i operator/(int32_t p_scalar) const;
	constexpr Vector3i &operator%=(int32_t p_scalar);
	constexpr Vector3i operator%(int32_t p_scalar) const;

	constexpr Vector3i operator-() const;

	constexpr bool operator==(const Vector3i &p_other) const;
	constexpr bool operator!=(const Vector3i &p_other) const;
	constexpr bool operator<(const Vector3i &p_other) const;
	constexpr bool operator<=(const Vector3i &p_other) const;
	constexpr bool operator>(const Vector3i &p_other) const;
	constexpr bool operator>=(const Vector3i &p_other) const;

	operator String() const;
	operator Vector3() const;

	constexpr Vector3i() :
			x(0), y(0), z(0) {}

	constexpr Vector3i(int32_t p_x, int32_t p_y, int32_t p_z) :
			x(p_x), y(p_y), z(p_z) {}
};

constexpr const int32_t &Vector3i::operator[](size_t p_axis) const {
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
		default:
			return coord[p_axis];
	}
}

constexpr int32_t &Vector3i::operator[](size_t p_axis) {
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
		default:
			return coord[p_axis];
	}
}

constexpr Vector3i &Vector3i::operator+=(const Vector3i &p_other) {
	x += p_other.x;
	y += p_other.y;
	z += p_other.z;
	return *this;
}

constexpr Vector3i Vector3i::operator+(const Vector3i &p_other) const {
	return Vector3i(x + p_other.x, y + p_other.y, z + p_other.z);
}

constexpr Vector3i &Vector3i::operator-=(const Vector3i &p_other) {
	x -= p_other.x;
	y -= p_other.y;
	z -= p_other.z;
	return *this;
}

constexpr Vector3i Vector3i::operator-(const Vector3i &p_other) const {
	return Vector3i(x - p_other.x, y - p_other.y, z - p_other.z);
}

constexpr Vector3i &Vector3i::operator*=(const Vector3i &p_other) {
	x *= p_other.x;
	y *= p_other.y;
	z *= p_other.z;
	return *this;
}

constexpr Vector3i Vector3i::operator*(const Vector3i &p_other) const {
	return Vector3i(x * p_other.x, y * p_other.y, z * p_other.z);
}

constexpr Vector3i &Vector3i::operator/=(const Vector3i &p_other) {
	x /= p_other.x;
	y /= p_other.y;
	z /= p_other.z;
	return *this;
}

constexpr Vector3i Vector3i::operator/(const Vector3i &p_other) const {
	return Vector3i(x / p_other.x, y / p_other.y, z / p_other.z);
}

constexpr Vector3i &Vector3i::operator%=(const Vector3i &p_other) {
	x %= p_other.x;
	y %= p_other.y;
	z %= p_other.z;
	return *this;
}

constexpr Vector3i Vector3i::operator%(const Vector3i &p_other) const {
	return Vector3i(x % p_other.x, y % p_other.y, z % p_other.z);
}

constexpr Vector3i &Vector3i::operator*=(int32_t p_scalar) {
	x *= p_scalar;
	y *= p_scalar;
	z *= p_scalar;
	return *this;
}

constexpr Vector3i Vector3i::operator*(int32_t p_scalar) const {
	return Vector3i(x * p_scalar, y * p_scalar, z * p_scalar);
}

constexpr Vector3i &Vector3i::operator/=(int32_t p_scalar) {
	x /= p_scalar;
	y /= p_scalar;
	z /= p_scalar;
	return *this;
}

constexpr Vector3i Vector3i::operator/(int32_t p_scalar) const {
	return Vector3i(x / p_scalar, y / p_scalar, z / p_scalar);
}

constexpr Vector3i &Vector3i::operator%=(int32_t p_scalar) {
	x %= p_scalar;
	y %= p_scalar;
	z %= p_scalar;
	return *this;
}

constexpr Vector3i Vector3i::operator%(int32_t p_scalar) const {
	return Vector3i(x % p_scalar, y % p_scalar, z % p_scalar);
}

constexpr Vector3i Vector3i::operator-() const {
	return Vector3i(-x, -y, -z);
}

constexpr bool Vector3i::operator==(const Vector3i &p_other) const {
	return (x == p_other.x && y == p_other.y && z == p_other.z);
}

constexpr bool Vector3i::operator!=(const Vector3i &p_other) const {
	return (x != p_other.x || y != p_other.y || z != p_other.z);
}

constexpr bool Vector3i::operator<(const Vector3i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			return z < p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector3i::operator>(const Vector3i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			return z > p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

constexpr bool Vector3i::operator<=(const Vector3i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			return z <= p_other.z;
		}
		return y < p_other.y;
	}
	return x < p_other.x;
}

constexpr bool Vector3i::operator>=(const Vector3i &p_other) const {
	if (x == p_other.x) {
		if (y == p_other.y) {
			return z >= p_other.z;
		}
		return y > p_other.y;
	}
	return x > p_other.x;
}

// Multiplication operators required to workaround issues with LLVM using implicit conversion.

constexpr Vector3i operator*(int32_t p_scalar, const Vector3i &p_vector3i) {
	return p_vector3i * p_scalar;
}

constexpr Vector3i operator*(int64_t p_scalar, const Vector3i &p_vector3i) {
	return p_vector3i * p_scalar;
}

constexpr Vector3i operator*(float p_scalar, const Vector3i &p_vector3i) {
	return p_vector3i * p_scalar;
}

constexpr Vector3i operator*(double p_scalar, const Vector3i &p_vector3i) {
	return p_vector3i * p_scalar;
}

int64_t Vector3i::length_squared() const {
	return x * (int64_t)x + y * (int64_t)y + z * (int64_t)z;
}

double Vector3i::length() const {
	return Math::sqrt((double)length_squared());
}

Vector3i Vector3i::abs() const {
	return Vector3i(Math::abs(x), Math::abs(y), Math::abs(z));
}

Vector3i Vector3i::sign() const {
	return Vector3i(SIGN(x), SIGN(y), SIGN(z));
}

double Vector3i::distance_to(const Vector3i &p_to) const {
	return (p_to - *this).length();
}

int64_t Vector3i::distance_squared_to(const Vector3i &p_to) const {
	return (p_to - *this).length_squared();
}

void Vector3i::zero() {
	x = y = z = 0;
}

#endif // VECTOR3I_H

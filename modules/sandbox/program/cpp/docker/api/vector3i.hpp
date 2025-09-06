/**************************************************************************/
/*  vector3i.hpp                                                          */
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

#pragma once
#include "syscalls_fwd.hpp"
#include <cmath>

struct Vector3i {
	int x;
	int y;
	int z;

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Vector3i, abs);
	METHOD(Vector3i, clamp);
	METHOD(Vector3i, clampi);
	METHOD(int, distance_squared_to);
	METHOD(real_t, distance_to);
	METHOD(real_t, length);
	METHOD(int, length_squared);
	METHOD(Vector3i, max);
	METHOD(int, max_axis_index);
	METHOD(Vector3i, maxi);
	METHOD(Vector3i, min);
	METHOD(int, min_axis_index);
	METHOD(Vector3i, mini);
	METHOD(Vector3i, sign);
	METHOD(Vector3i, snapped);
	METHOD(Vector3i, snappedi);

	Vector3i &operator+=(const Vector3i &other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}
	Vector3i &operator-=(const Vector3i &other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}
	Vector3i &operator*=(const Vector3i &other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}
	Vector3i &operator/=(const Vector3i &other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		return *this;
	}

	bool operator==(const Vector3i &other) const {
		return __builtin_memcmp(this, &other, sizeof(Vector3i)) == 0;
	}
	bool operator!=(const Vector3i &other) const {
		return !(*this == other);
	}

	constexpr Vector3i() :
			x(0), y(0), z(0) {}
	constexpr Vector3i(int val) :
			x(val), y(val), z(val) {}
	constexpr Vector3i(int x, int y, int z) :
			x(x), y(y), z(z) {}
};

inline constexpr auto operator+(const Vector3i &a, const Vector3i &b) noexcept {
	return Vector3i{ a.x + b.x, a.y + b.y, a.z + b.z };
}
inline constexpr auto operator-(const Vector3i &a, const Vector3i &b) noexcept {
	return Vector3i{ a.x - b.x, a.y - b.y, a.z - b.z };
}
inline constexpr auto operator*(const Vector3i &a, const Vector3i &b) noexcept {
	return Vector3i{ a.x * b.x, a.y * b.y, a.z * b.z };
}
inline constexpr auto operator/(const Vector3i &a, const Vector3i &b) noexcept {
	return Vector3i{ a.x / b.x, a.y / b.y, a.z / b.z };
}

inline constexpr auto operator+(const Vector3i &a, int b) noexcept {
	return Vector3i{ a.x + b, a.y + b, a.z + b };
}
inline constexpr auto operator-(const Vector3i &a, int b) noexcept {
	return Vector3i{ a.x - b, a.y - b, a.z - b };
}
inline constexpr auto operator*(const Vector3i &a, int b) noexcept {
	return Vector3i{ a.x * b, a.y * b, a.z * b };
}
inline constexpr auto operator/(const Vector3i &a, int b) noexcept {
	return Vector3i{ a.x / b, a.y / b, a.z / b };
}

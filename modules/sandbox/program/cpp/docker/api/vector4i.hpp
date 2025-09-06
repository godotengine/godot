/**************************************************************************/
/*  vector4i.hpp                                                          */
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

struct Vector4i {
	int x;
	int y;
	int z;
	int w;

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Vector4i, abs);
	METHOD(Vector4i, clamp);
	METHOD(Vector4i, clampi);
	METHOD(int, distance_squared_to);
	METHOD(real_t, distance_to);
	METHOD(real_t, length);
	METHOD(int, length_squared);
	METHOD(Vector4i, max);
	METHOD(int, max_axis_index);
	METHOD(Vector4i, maxi);
	METHOD(Vector4i, min);
	METHOD(int, min_axis_index);
	METHOD(Vector4i, mini);
	METHOD(Vector4i, sign);
	METHOD(Vector4i, snapped);
	METHOD(Vector4i, snappedi);

	Vector4i &operator+=(const Vector4i &other) {
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;
		return *this;
	}
	Vector4i &operator-=(const Vector4i &other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		w -= other.w;
		return *this;
	}
	Vector4i &operator*=(const Vector4i &other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		w *= other.w;
		return *this;
	}
	Vector4i &operator/=(const Vector4i &other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		w /= other.w;
		return *this;
	}

	bool operator==(const Vector4i &other) const {
		return __builtin_memcmp(this, &other, sizeof(Vector4i)) == 0;
	}
	bool operator!=(const Vector4i &other) const {
		return !(*this == other);
	}

	constexpr Vector4i() :
			x(0), y(0), z(0), w(0) {}
	constexpr Vector4i(int val) :
			x(val), y(val), z(val), w(val) {}
	constexpr Vector4i(int x, int y, int z, int w) :
			x(x), y(y), z(z), w(w) {}
};

inline constexpr auto operator+(const Vector4i &a, const Vector4i &b) noexcept {
	return Vector4i{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}
inline constexpr auto operator-(const Vector4i &a, const Vector4i &b) noexcept {
	return Vector4i{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}
inline constexpr auto operator*(const Vector4i &a, const Vector4i &b) noexcept {
	return Vector4i{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}
inline constexpr auto operator/(const Vector4i &a, const Vector4i &b) noexcept {
	return Vector4i{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

inline constexpr auto operator+(const Vector4i &a, int b) noexcept {
	return Vector4i{ a.x + b, a.y + b, a.z + b, a.w + b };
}
inline constexpr auto operator-(const Vector4i &a, int b) noexcept {
	return Vector4i{ a.x - b, a.y - b, a.z - b, a.w - b };
}
inline constexpr auto operator*(const Vector4i &a, int b) noexcept {
	return Vector4i{ a.x * b, a.y * b, a.z * b, a.w * b };
}
inline constexpr auto operator/(const Vector4i &a, int b) noexcept {
	return Vector4i{ a.x / b, a.y / b, a.z / b, a.w / b };
}

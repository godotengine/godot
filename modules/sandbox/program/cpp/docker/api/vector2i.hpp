/**************************************************************************/
/*  vector2i.hpp                                                          */
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
#include <string_view>

struct Vector2i {
	int x;
	int y;

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	Vector2i &operator+=(const Vector2i &other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	Vector2i &operator-=(const Vector2i &other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	Vector2i &operator*=(const Vector2i &other) {
		x *= other.x;
		y *= other.y;
		return *this;
	}
	Vector2i &operator/=(const Vector2i &other) {
		x /= other.x;
		y /= other.y;
		return *this;
	}

	bool operator==(const Vector2i &other) const {
		return x == other.x && y == other.y;
	}
	bool operator!=(const Vector2i &other) const {
		return !this->operator==(other);
	}

	constexpr Vector2i() :
			x(0), y(0) {}
	constexpr Vector2i(int val) :
			x(val), y(val) {}
	constexpr Vector2i(int x, int y) :
			x(x), y(y) {}
};

inline constexpr auto operator+(const Vector2i &a, const Vector2i &b) noexcept {
	return Vector2i{ a.x + b.x, a.y + b.y };
}
inline constexpr auto operator-(const Vector2i &a, const Vector2i &b) noexcept {
	return Vector2i{ a.x - b.x, a.y - b.y };
}
inline constexpr auto operator*(const Vector2i &a, const Vector2i &b) noexcept {
	return Vector2i{ a.x * b.x, a.y * b.y };
}
inline constexpr auto operator/(const Vector2i &a, const Vector2i &b) noexcept {
	return Vector2i{ a.x / b.x, a.y / b.y };
}

inline constexpr auto operator+(const Vector2i &a, int b) noexcept {
	return Vector2i{ a.x + b, a.y + b };
}
inline constexpr auto operator-(const Vector2i &a, int b) noexcept {
	return Vector2i{ a.x - b, a.y - b };
}
inline constexpr auto operator*(const Vector2i &a, int b) noexcept {
	return Vector2i{ a.x * b, a.y * b };
}
inline constexpr auto operator/(const Vector2i &a, int b) noexcept {
	return Vector2i{ a.x / b, a.y / b };
}

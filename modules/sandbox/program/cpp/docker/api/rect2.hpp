/**************************************************************************/
/*  rect2.hpp                                                             */
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
#include "vector2.hpp"

struct Rect2 {
	Vector2 position;
	Vector2 size;

	const Vector2 &get_position() const { return position; }
	void set_position(const Vector2 &p_pos) { position = p_pos; }
	const Vector2 &get_size() const { return size; }
	void set_size(const Vector2 &p_size) { size = p_size; }
	Vector2 get_end() const { return position + size; }
	void set_end(const Vector2 &p_end) { size = p_end - position; }

	real_t get_area() const { return size.x * size.y; }
	Vector2 get_center() const { return position + size * real_t(0.5); }

	bool has_area() const { return size.x > 0 && size.y > 0; }

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Rect2, abs);
	METHOD(bool, encloses);
	METHOD(Rect2, expand);
	METHOD(Rect2, grow);
	METHOD(Rect2, grow_individual);
	METHOD(Rect2, grow_side);
	METHOD(bool, has_point);
	METHOD(Rect2, intersection);
	METHOD(bool, intersects);
	METHOD(bool, is_equal_approx);
	METHOD(bool, is_finite);
	METHOD(Rect2, merge);

	bool operator==(const Rect2 &other) const {
		return __builtin_memcmp(this, &other, sizeof(Rect2)) == 0;
	}
	bool operator!=(const Rect2 &other) const {
		return !this->operator==(other);
	}

	constexpr Rect2() :
			position(), size() {}
	constexpr Rect2(Vector2 position, Vector2 size) :
			position(position), size(size) {}
	constexpr Rect2(real_t x, real_t y, real_t width, real_t height) :
			position(x, y), size(width, height) {}
};

inline constexpr auto operator+(const Rect2 &a, const Vector2 &b) noexcept {
	return Rect2{ a.position + b, a.size };
}
inline constexpr auto operator-(const Rect2 &a, const Vector2 &b) noexcept {
	return Rect2{ a.position - b, a.size };
}
inline constexpr auto operator*(const Rect2 &a, const Vector2 &b) noexcept {
	return Rect2{ a.position * b, a.size * b };
}
inline constexpr auto operator/(const Rect2 &a, const Vector2 &b) noexcept {
	return Rect2{ a.position / b, a.size / b };
}

inline constexpr auto operator+(const Rect2 &a, real_t b) noexcept {
	return Rect2{ a.position + b, a.size };
}
inline constexpr auto operator-(const Rect2 &a, real_t b) noexcept {
	return Rect2{ a.position - b, a.size };
}
inline constexpr auto operator*(const Rect2 &a, real_t b) noexcept {
	return Rect2{ a.position * b, a.size * b };
}
inline constexpr auto operator/(const Rect2 &a, real_t b) noexcept {
	return Rect2{ a.position / b, a.size / b };
}

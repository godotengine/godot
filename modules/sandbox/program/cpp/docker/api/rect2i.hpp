/**************************************************************************/
/*  rect2i.hpp                                                            */
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
#include "vector2i.hpp"

struct Rect2i {
	Vector2i position;
	Vector2i size;

	const Vector2i &get_position() const { return position; }
	void set_position(Vector2i p_position) { position = p_position; }
	const Vector2i &get_size() const { return size; }
	void set_size(Vector2i p_size) { size = p_size; }
	Vector2i get_end() const { return position + size; }
	void set_end(Vector2i p_end) { size = p_end - position; }

	int get_area() const { return size.x * size.y; }
	bool has_area() const { return size.x > 0 && size.y > 0; }
	Vector2i get_center() const { return position + size / 2; }

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Rect2i, abs);
	METHOD(bool, encloses);
	METHOD(Rect2i, expand);
	METHOD(Rect2i, grow);
	METHOD(Rect2i, grow_individual);
	METHOD(Rect2i, grow_side);
	METHOD(bool, has_point);
	METHOD(Rect2i, intersection);
	METHOD(bool, intersects);
	METHOD(bool, merge);

	bool operator==(const Rect2i &other) const {
		return __builtin_memcmp(this, &other, sizeof(Rect2i)) == 0;
	}
	bool operator!=(const Rect2i &other) const {
		return !this->operator==(other);
	}

	constexpr Rect2i() :
			position(), size() {}
	constexpr Rect2i(Vector2i position, Vector2i size) :
			position(position), size(size) {}
	constexpr Rect2i(int x, int y, int width, int height) :
			position(x, y), size(width, height) {}
};

inline constexpr auto operator+(const Rect2i &a, const Vector2i &b) noexcept {
	return Rect2i{ a.position + b, a.size };
}
inline constexpr auto operator-(const Rect2i &a, const Vector2i &b) noexcept {
	return Rect2i{ a.position - b, a.size };
}
inline constexpr auto operator*(const Rect2i &a, const Vector2i &b) noexcept {
	return Rect2i{ a.position * b, a.size * b };
}
inline constexpr auto operator/(const Rect2i &a, const Vector2i &b) noexcept {
	return Rect2i{ a.position / b, a.size / b };
}

inline constexpr auto operator+(const Rect2i &a, int b) noexcept {
	return Rect2i{ a.position + b, a.size };
}
inline constexpr auto operator-(const Rect2i &a, int b) noexcept {
	return Rect2i{ a.position - b, a.size };
}
inline constexpr auto operator*(const Rect2i &a, int b) noexcept {
	return Rect2i{ a.position * b, a.size * b };
}
inline constexpr auto operator/(const Rect2i &a, int b) noexcept {
	return Rect2i{ a.position / b, a.size / b };
}

/**************************************************************************/
/*  vector3.hpp                                                           */
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

struct Vector3 {
	real_t x;
	real_t y;
	real_t z;

	float length() const noexcept;
	float length_squared() const noexcept { return this->dot(*this); }
	void normalize() { *this = normalized(); }
	Vector3 normalized() const noexcept;
	float dot(const Vector3 &other) const noexcept;
	Vector3 cross(const Vector3 &other) const noexcept;
	float distance_to(const Vector3 &other) const noexcept;
	float distance_squared_to(const Vector3 &other) const noexcept;
	float angle_to(const Vector3 &other) const noexcept;
	Vector3 direction_to(const Vector3 &other) const noexcept;
	Vector3 floor() const noexcept;

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Vector3, abs);
	METHOD(Vector3, bezier_derivative);
	METHOD(Vector3, bezier_interpolate);
	METHOD(Vector3, bounce);
	METHOD(Vector3, ceil);
	METHOD(Vector3, clamp);
	METHOD(Vector3, clampf);
	METHOD(Vector3, cubic_interpolate);
	METHOD(Vector3, cubic_interpolate_in_time);
	METHOD(Vector3, inverse);
	METHOD(bool, is_equal_approx);
	METHOD(bool, is_finite);
	METHOD(bool, is_normalized);
	METHOD(bool, is_zero_approx);
	METHOD(Vector3, lerp);
	METHOD(Vector3, limit_length);
	METHOD(Vector3, max);
	METHOD(int, max_axis_index);
	METHOD(Vector3, maxf);
	METHOD(Vector3, min);
	METHOD(int, min_axis_index);
	METHOD(Vector3, minf);
	METHOD(Vector3, move_toward);
	METHOD(Vector3, octahedron_decode);
	//METHOD(Vector2, octahedron_encode);
	//METHOD(Basis,   outer);
	METHOD(Vector3, posmod);
	METHOD(Vector3, posmodv);
	METHOD(Vector3, project);
	METHOD(Vector3, reflect);
	METHOD(Vector3, rotated);
	METHOD(Vector3, round);
	METHOD(Vector3, sign);
	METHOD(real_t, signed_angle_to);
	METHOD(Vector3, slerp);
	METHOD(Vector3, slide);
	METHOD(Vector3, snapped);
	METHOD(Vector3, snappedf);

	Vector3 &operator+=(const Vector3 &other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}
	Vector3 &operator-=(const Vector3 &other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}
	Vector3 &operator*=(const Vector3 &other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}
	Vector3 &operator/=(const Vector3 &other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		return *this;
	}

	bool operator==(const Vector3 &other) const {
		return __builtin_memcmp(this, &other, sizeof(Vector3)) == 0;
	}
	bool operator!=(const Vector3 &other) const {
		return !(*this == other);
	}

	constexpr Vector3() :
			x(0), y(0), z(0) {}
	constexpr Vector3(real_t val) :
			x(val), y(val), z(val) {}
	constexpr Vector3(real_t x, real_t y, real_t z) :
			x(x), y(y), z(z) {}

	static Vector3 const ZERO;
	static Vector3 const ONE;
	static Vector3 const LEFT;
	static Vector3 const RIGHT;
	static Vector3 const UP;
	static Vector3 const DOWN;
	static Vector3 const FORWARD;
	static Vector3 const BACK;
};
inline constexpr Vector3 const Vector3::ZERO = Vector3(0, 0, 0);
inline constexpr Vector3 const Vector3::ONE = Vector3(1, 1, 1);
inline constexpr Vector3 const Vector3::LEFT = Vector3(-1, 0, 0);
inline constexpr Vector3 const Vector3::RIGHT = Vector3(1, 0, 0);
inline constexpr Vector3 const Vector3::UP = Vector3(0, 1, 0);
inline constexpr Vector3 const Vector3::DOWN = Vector3(0, -1, 0);
inline constexpr Vector3 const Vector3::FORWARD = Vector3(0, 0, -1);
inline constexpr Vector3 const Vector3::BACK = Vector3(0, 0, 1);

inline constexpr auto operator+(const Vector3 &a, const Vector3 &b) noexcept {
	return Vector3{ a.x + b.x, a.y + b.y, a.z + b.z };
}
inline constexpr auto operator-(const Vector3 &a, const Vector3 &b) noexcept {
	return Vector3{ a.x - b.x, a.y - b.y, a.z - b.z };
}
inline constexpr auto operator*(const Vector3 &a, const Vector3 &b) noexcept {
	return Vector3{ a.x * b.x, a.y * b.y, a.z * b.z };
}
inline constexpr auto operator/(const Vector3 &a, const Vector3 &b) noexcept {
	return Vector3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

inline constexpr auto operator+(const Vector3 &a, real_t b) noexcept {
	return Vector3{ a.x + b, a.y + b, a.z + b };
}
inline constexpr auto operator-(const Vector3 &a, real_t b) noexcept {
	return Vector3{ a.x - b, a.y - b, a.z - b };
}
inline constexpr auto operator*(const Vector3 &a, real_t b) noexcept {
	return Vector3{ a.x * b, a.y * b, a.z * b };
}
inline constexpr auto operator/(const Vector3 &a, real_t b) noexcept {
	return Vector3{ a.x / b, a.y / b, a.z / b };
}

inline Vector3 Vector3::floor() const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register float resultX asm("fa0");
	register float resultY asm("fa1");
	register float resultZ asm("fa2");
	register int op asm("a2") = 11; // Vec3_Op::FLOOR
	register int syscall asm("a7") = 537; // ECALL_VEC3_OPS

	__asm__ volatile("ecall"
			: "=f"(resultX), "=f"(resultY), "=f"(resultZ)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(syscall));
	return { resultX, resultY, resultZ };
}

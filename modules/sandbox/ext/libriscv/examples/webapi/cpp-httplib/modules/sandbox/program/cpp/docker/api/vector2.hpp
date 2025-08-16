#pragma once
#include <cmath>
#include <string_view>
#include "syscalls_fwd.hpp"

struct Vector2 {
	real_t x;
	real_t y;

	float length() const noexcept;
	float length_squared() const noexcept { return x * x + y * y; }
	Vector2 limit_length(double length) const noexcept;

	void normalize() { *this = normalized(); }
	Vector2 normalized() const noexcept;
	float distance_to(const Vector2& other) const noexcept;
	Vector2 direction_to(const Vector2& other) const noexcept;
	float dot(const Vector2& other) const noexcept;
	static Vector2 sincos(float angle) noexcept;
	static Vector2 from_angle(float angle) noexcept;

	Vector2 lerp(const Vector2& to, double weight) const noexcept;
	Vector2 cubic_interpolate(const Vector2& b, const Vector2& pre_a, const Vector2& post_b, double weight) const noexcept;
	Vector2 slerp(const Vector2& to, double weight) const noexcept;

	Vector2 slide(const Vector2& normal) const noexcept;
	Vector2 bounce(const Vector2& normal) const noexcept;
	Vector2 reflect(const Vector2& normal) const noexcept;

	void rotate(real_t angle) noexcept { *this = rotated(angle); }
	Vector2 rotated(real_t angle) const noexcept;

	Vector2 project(const Vector2& vec) const noexcept;
	Vector2 orthogonal() const noexcept { return {y, -x}; }
	float aspect() const noexcept { return x / y; }

	real_t operator [] (int index) const {
		return index == 0 ? x : y;
	}
	real_t& operator [] (int index) {
		return index == 0 ? x : y;
	}

	METHOD(Vector2, abs);
	METHOD(Vector2, bezier_derivative);
	METHOD(Vector2, bezier_interpolate);
	METHOD(Vector2, ceil);
	METHOD(Vector2, clamp);
	METHOD(Vector2, clampf);
	METHOD(real_t,  cross);
	METHOD(Vector2, cubic_interpolate_in_time);
	METHOD(Vector2, floor);
	METHOD(bool,    is_equal_approx);
	METHOD(bool,    is_finite);
	METHOD(bool,    is_normalized);
	METHOD(bool,    is_zero_approx);
	METHOD(Vector2, max);
	METHOD(Vector2, maxf);
	METHOD(int,     max_axis_index);
	METHOD(Vector2, min);
	METHOD(Vector2, minf);
	METHOD(int,     min_axis_index);
	METHOD(Vector2, move_toward);
	METHOD(Vector2, posmod);
	METHOD(Vector2, posmodv);
	METHOD(Vector2, round);
	METHOD(Vector2, sign);
	METHOD(Vector2, snapped);
	METHOD(Vector2, snappedf);

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	constexpr Vector2& operator += (const Vector2& other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	constexpr Vector2& operator -= (const Vector2& other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	constexpr Vector2& operator *= (const Vector2& other) {
		x *= other.x;
		y *= other.y;
		return *this;
	}
	constexpr Vector2& operator /= (const Vector2& other) {
		x /= other.x;
		y /= other.y;
		return *this;
	}

	bool operator == (const Vector2& other) const {
		return x == other.x && y == other.y;
	}

	constexpr Vector2() : x(0), y(0) {}
	constexpr Vector2(real_t val) : x(val), y(val) {}
	constexpr Vector2(real_t x, real_t y) : x(x), y(y) {}

	static Vector2 const ZERO;
	static Vector2 const ONE;
	static Vector2 const LEFT;
	static Vector2 const RIGHT;
	static Vector2 const UP;
	static Vector2 const DOWN;
};
inline constexpr Vector2 const Vector2::ZERO = Vector2(0, 0);
inline constexpr Vector2 const Vector2::ONE = Vector2(1, 1);
inline constexpr Vector2 const Vector2::LEFT = Vector2(-1, 0);
inline constexpr Vector2 const Vector2::RIGHT = Vector2(1, 0);
inline constexpr Vector2 const Vector2::UP = Vector2(0, -1);
inline constexpr Vector2 const Vector2::DOWN = Vector2(0, 1);


inline constexpr auto operator + (const Vector2& a, const Vector2& b) noexcept {
	return Vector2{a.x + b.x, a.y + b.y};
}
inline constexpr auto operator - (const Vector2& a, const Vector2& b) noexcept {
	return Vector2{a.x - b.x, a.y - b.y};
}
inline constexpr auto operator * (const Vector2& a, const Vector2& b) noexcept {
	return Vector2{a.x * b.x, a.y * b.y};
}
inline constexpr auto operator / (const Vector2& a, const Vector2& b) noexcept {
	return Vector2{a.x / b.x, a.y / b.y};
}

inline constexpr auto operator + (const Vector2& a, real_t b) noexcept {
	return Vector2{a.x + b, a.y + b};
}
inline constexpr auto operator - (const Vector2& a, real_t b) noexcept {
	return Vector2{a.x - b, a.y - b};
}
inline constexpr auto operator * (const Vector2& a, real_t b) noexcept {
	return Vector2{a.x * b, a.y * b};
}
inline constexpr auto operator / (const Vector2& a, real_t b) noexcept {
	return Vector2{a.x / b, a.y / b};
}

inline float Vector2::length() const noexcept {
	register float x asm("fa0") = this->x;
	register float y asm("fa1") = this->y;
	register int syscall asm("a7") = 514; // ECALL_VEC2_LENGTH

	__asm__ volatile("ecall"
					 : "+f"(x)
					 : "f"(y), "r"(syscall));
	return x;
}

inline Vector2 Vector2::normalized() const noexcept {
	register float x asm("fa0") = this->x;
	register float y asm("fa1") = this->y;
	register int syscall asm("a7") = 515; // ECALL_VEC2_NORMALIZED

	__asm__ volatile("ecall"
					 : "+f"(x), "+f"(y)
					 : "r"(syscall));
	return {x, y};
}

inline Vector2 Vector2::rotated(real_t angle) const noexcept {
	register float x asm("fa0") = this->x;
	register float y asm("fa1") = this->y;
	register float a asm("fa2") = angle;
	register int syscall asm("a7") = 516; // ECALL_VEC2_ROTATED

	__asm__ volatile("ecall"
					 : "+f"(x), "+f"(y)
					 : "f"(a), "r"(syscall));
	return {x, y};
}

inline float Vector2::distance_to(const Vector2& other) const noexcept {
	return (*this - other).length();
}
inline Vector2 Vector2::direction_to(const Vector2& other) const noexcept {
	return (*this - other).normalized();
}
inline float Vector2::dot(const Vector2& other) const noexcept {
	return x * other.x + y * other.y;
}
inline Vector2 Vector2::sincos(float angle) noexcept {
	register float s asm("fa0") = angle;
	register float c asm("fa1");
	register int syscall asm("a7") = 513; // ECALL_SINCOS

	__asm__ volatile("ecall"
					 : "+f"(s), "=f"(c)
					 : "r"(syscall));
	return {s, c}; // (sine, cosine)
}
inline Vector2 Vector2::from_angle(float angle) noexcept {
	Vector2 v = sincos(angle);
	return {v.y, v.x}; // (cos(angle), sin(angle))
}

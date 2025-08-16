#pragma once
#include <cmath>
#include <string_view>
#include "syscalls_fwd.hpp"

struct Vector2i {
	int x;
	int y;

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	Vector2i& operator += (const Vector2i& other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	Vector2i& operator -= (const Vector2i& other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	Vector2i& operator *= (const Vector2i& other) {
		x *= other.x;
		y *= other.y;
		return *this;
	}
	Vector2i& operator /= (const Vector2i& other) {
		x /= other.x;
		y /= other.y;
		return *this;
	}

	bool operator == (const Vector2i& other) const {
		return x == other.x && y == other.y;
	}
	bool operator != (const Vector2i& other) const {
		return !this->operator==(other);
	}

	constexpr Vector2i() : x(0), y(0) {}
	constexpr Vector2i(int val) : x(val), y(val) {}
	constexpr Vector2i(int x, int y) : x(x), y(y) {}
};

inline constexpr auto operator + (const Vector2i& a, const Vector2i& b) noexcept {
	return Vector2i{a.x + b.x, a.y + b.y};
}
inline constexpr auto operator - (const Vector2i& a, const Vector2i& b) noexcept {
	return Vector2i{a.x - b.x, a.y - b.y};
}
inline constexpr auto operator * (const Vector2i& a, const Vector2i& b) noexcept {
	return Vector2i{a.x * b.x, a.y * b.y};
}
inline constexpr auto operator / (const Vector2i& a, const Vector2i& b) noexcept {
	return Vector2i{a.x / b.x, a.y / b.y};
}

inline constexpr auto operator + (const Vector2i& a, int b) noexcept {
	return Vector2i{a.x + b, a.y + b};
}
inline constexpr auto operator - (const Vector2i& a, int b) noexcept {
	return Vector2i{a.x - b, a.y - b};
}
inline constexpr auto operator * (const Vector2i& a, int b) noexcept {
	return Vector2i{a.x * b, a.y * b};
}
inline constexpr auto operator / (const Vector2i& a, int b) noexcept {
	return Vector2i{a.x / b, a.y / b};
}

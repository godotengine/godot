#pragma once
#include <cmath>
#include "syscalls_fwd.hpp"

struct Vector3i {
	int x;
	int y;
	int z;

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Vector3i, abs);
	METHOD(Vector3i, clamp);
	METHOD(Vector3i, clampi);
	METHOD(int,      distance_squared_to);
	METHOD(real_t,   distance_to);
	METHOD(real_t,   length);
	METHOD(int,      length_squared);
	METHOD(Vector3i, max);
	METHOD(int,      max_axis_index);
	METHOD(Vector3i, maxi);
	METHOD(Vector3i, min);
	METHOD(int,      min_axis_index);
	METHOD(Vector3i, mini);
	METHOD(Vector3i, sign);
	METHOD(Vector3i, snapped);
	METHOD(Vector3i, snappedi);

	Vector3i& operator += (const Vector3i& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}
	Vector3i& operator -= (const Vector3i& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}
	Vector3i& operator *= (const Vector3i& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}
	Vector3i& operator /= (const Vector3i& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		return *this;
	}

	bool operator == (const Vector3i& other) const {
		return __builtin_memcmp(this, &other, sizeof(Vector3i)) == 0;
	}
	bool operator != (const Vector3i& other) const {
		return !(*this == other);
	}

	constexpr Vector3i() : x(0), y(0), z(0) {}
	constexpr Vector3i(int val) : x(val), y(val), z(val) {}
	constexpr Vector3i(int x, int y, int z) : x(x), y(y), z(z) {}
};

inline constexpr auto operator + (const Vector3i& a, const Vector3i& b) noexcept {
	return Vector3i{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline constexpr auto operator - (const Vector3i& a, const Vector3i& b) noexcept {
	return Vector3i{a.x - b.x, a.y - b.y, a.z - b.z};
}
inline constexpr auto operator * (const Vector3i& a, const Vector3i& b) noexcept {
	return Vector3i{a.x * b.x, a.y * b.y, a.z * b.z};
}
inline constexpr auto operator / (const Vector3i& a, const Vector3i& b) noexcept {
	return Vector3i{a.x / b.x, a.y / b.y, a.z / b.z};
}

inline constexpr auto operator + (const Vector3i& a, int b) noexcept {
	return Vector3i{a.x + b, a.y + b, a.z + b};
}
inline constexpr auto operator - (const Vector3i& a, int b) noexcept {
	return Vector3i{a.x - b, a.y - b, a.z - b};
}
inline constexpr auto operator * (const Vector3i& a, int b) noexcept {
	return Vector3i{a.x * b, a.y * b, a.z * b};
}
inline constexpr auto operator / (const Vector3i& a, int b) noexcept {
	return Vector3i{a.x / b, a.y / b, a.z / b};
}

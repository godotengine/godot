#pragma once
#include <cmath>
#include "syscalls_fwd.hpp"

struct Vector4i {
	int x;
	int y;
	int z;
	int w;

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Vector4i, abs);
	METHOD(Vector4i, clamp);
	METHOD(Vector4i, clampi);
	METHOD(int,      distance_squared_to);
	METHOD(real_t,   distance_to);
	METHOD(real_t,   length);
	METHOD(int,      length_squared);
	METHOD(Vector4i, max);
	METHOD(int,      max_axis_index);
	METHOD(Vector4i, maxi);
	METHOD(Vector4i, min);
	METHOD(int,      min_axis_index);
	METHOD(Vector4i, mini);
	METHOD(Vector4i, sign);
	METHOD(Vector4i, snapped);
	METHOD(Vector4i, snappedi);

	Vector4i& operator += (const Vector4i& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;
		return *this;
	}
	Vector4i& operator -= (const Vector4i& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		w -= other.w;
		return *this;
	}
	Vector4i& operator *= (const Vector4i& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		w *= other.w;
		return *this;
	}
	Vector4i& operator /= (const Vector4i& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		w /= other.w;
		return *this;
	}

	bool operator == (const Vector4i& other) const {
		return __builtin_memcmp(this, &other, sizeof(Vector4i)) == 0;
	}
	bool operator != (const Vector4i& other) const {
		return !(*this == other);
	}

	constexpr Vector4i() : x(0), y(0), z(0), w(0) {}
	constexpr Vector4i(int val) : x(val), y(val), z(val), w(val) {}
	constexpr Vector4i(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}
};

inline constexpr auto operator + (const Vector4i& a, const Vector4i& b) noexcept {
	return Vector4i{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline constexpr auto operator - (const Vector4i& a, const Vector4i& b) noexcept {
	return Vector4i{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
inline constexpr auto operator * (const Vector4i& a, const Vector4i& b) noexcept {
	return Vector4i{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
inline constexpr auto operator / (const Vector4i& a, const Vector4i& b) noexcept {
	return Vector4i{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

inline constexpr auto operator + (const Vector4i& a, int b) noexcept {
	return Vector4i{a.x + b, a.y + b, a.z + b, a.w + b};
}
inline constexpr auto operator - (const Vector4i& a, int b) noexcept {
	return Vector4i{a.x - b, a.y - b, a.z - b, a.w - b};
}
inline constexpr auto operator * (const Vector4i& a, int b) noexcept {
	return Vector4i{a.x * b, a.y * b, a.z * b, a.w * b};
}
inline constexpr auto operator / (const Vector4i& a, int b) noexcept {
	return Vector4i{a.x / b, a.y / b, a.z / b, a.w / b};
}

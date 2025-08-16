#pragma once
#include "vector2.hpp"

struct Rect2 {
	Vector2 position;
	Vector2 size;

	const Vector2& get_position() const { return position; }
	void set_position(const Vector2& p_pos) { position = p_pos; }
	const Vector2& get_size() const { return size; }
	void set_size(const Vector2& p_size) { size = p_size; }
	Vector2 get_end() const { return position + size; }
	void set_end(const Vector2& p_end) { size = p_end - position; }

	real_t get_area() const { return size.x * size.y; }
	Vector2 get_center() const { return position + size * real_t(0.5); }

	bool has_area() const { return size.x > 0 && size.y > 0; }

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Rect2, abs);
	METHOD(bool,  encloses);
	METHOD(Rect2, expand);
	METHOD(Rect2, grow);
	METHOD(Rect2, grow_individual);
	METHOD(Rect2, grow_side);
	METHOD(bool,  has_point);
	METHOD(Rect2, intersection);
	METHOD(bool,  intersects);
	METHOD(bool,  is_equal_approx);
	METHOD(bool,  is_finite);
	METHOD(Rect2, merge);

	bool operator == (const Rect2& other) const {
		return __builtin_memcmp(this, &other, sizeof(Rect2)) == 0;
	}
	bool operator != (const Rect2& other) const {
		return !this->operator==(other);
	}

	constexpr Rect2() : position(), size() {}
	constexpr Rect2(Vector2 position, Vector2 size) : position(position), size(size) {}
	constexpr Rect2(real_t x, real_t y, real_t width, real_t height) : position(x, y), size(width, height) {}
};

inline constexpr auto operator + (const Rect2& a, const Vector2& b) noexcept {
	return Rect2{a.position + b, a.size};
}
inline constexpr auto operator - (const Rect2& a, const Vector2& b) noexcept {
	return Rect2{a.position - b, a.size};
}
inline constexpr auto operator * (const Rect2& a, const Vector2& b) noexcept {
	return Rect2{a.position * b, a.size * b};
}
inline constexpr auto operator / (const Rect2& a, const Vector2& b) noexcept {
	return Rect2{a.position / b, a.size / b};
}

inline constexpr auto operator + (const Rect2& a, real_t b) noexcept {
	return Rect2{a.position + b, a.size};
}
inline constexpr auto operator - (const Rect2& a, real_t b) noexcept {
	return Rect2{a.position - b, a.size};
}
inline constexpr auto operator * (const Rect2& a, real_t b) noexcept {
	return Rect2{a.position * b, a.size * b};
}
inline constexpr auto operator / (const Rect2& a, real_t b) noexcept {
	return Rect2{a.position / b, a.size / b};
}

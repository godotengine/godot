#ifndef VECTOR2_H
#define VECTOR2_H

#include <gdnative/vector2.h>

#include "Defs.hpp"

#include <Math.hpp>

namespace godot {

class String;

struct Vector2 {

	union {
		real_t x;
		real_t width;
	};
	union {
		real_t y;
		real_t height;
	};

	inline Vector2(real_t p_x, real_t p_y) {
		x = p_x;
		y = p_y;
	}

	inline Vector2() {
		x = 0;
		y = 0;
	}

	inline real_t &operator[](int p_idx) {
		return p_idx ? y : x;
	}

	inline const real_t &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	inline Vector2 operator+(const Vector2 &p_v) const {
		return Vector2(x + p_v.x, y + p_v.y);
	}

	inline void operator+=(const Vector2 &p_v) {
		x += p_v.x;
		y += p_v.y;
	}

	inline Vector2 operator-(const Vector2 &p_v) const {
		return Vector2(x - p_v.x, y - p_v.y);
	}

	inline void operator-=(const Vector2 &p_v) {
		x -= p_v.x;
		y -= p_v.y;
	}

	inline Vector2 operator*(const Vector2 &p_v1) const {
		return Vector2(x * p_v1.x, y * p_v1.y);
	}

	inline Vector2 operator*(const real_t &rvalue) const {
		return Vector2(x * rvalue, y * rvalue);
	}

	inline void operator*=(const real_t &rvalue) {
		x *= rvalue;
		y *= rvalue;
	}

	inline void operator*=(const Vector2 &rvalue) {
		*this = *this * rvalue;
	}

	inline Vector2 operator/(const Vector2 &p_v1) const {
		return Vector2(x / p_v1.x, y / p_v1.y);
	}

	inline Vector2 operator/(const real_t &rvalue) const {
		return Vector2(x / rvalue, y / rvalue);
	}

	inline void operator/=(const real_t &rvalue) {
		x /= rvalue;
		y /= rvalue;
	}

	inline Vector2 operator-() const {
		return Vector2(-x, -y);
	}

	bool operator==(const Vector2 &p_vec2) const;

	bool operator!=(const Vector2 &p_vec2) const;

	inline bool operator<(const Vector2 &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	inline bool operator<=(const Vector2 &p_vec2) const { return (x == p_vec2.x) ? (y <= p_vec2.y) : (x <= p_vec2.x); }

	inline void normalize() {
		real_t l = x * x + y * y;
		if (l != 0) {
			l = sqrt(l);
			x /= l;
			y /= l;
		}
	}

	inline Vector2 normalized() const {
		Vector2 v = *this;
		v.normalize();
		return v;
	}

	inline real_t length() const {
		return sqrt(x * x + y * y);
	}

	inline real_t length_squared() const {
		return x * x + y * y;
	}

	inline real_t distance_to(const Vector2 &p_vector2) const {
		return sqrt((x - p_vector2.x) * (x - p_vector2.x) + (y - p_vector2.y) * (y - p_vector2.y));
	}

	inline real_t distance_squared_to(const Vector2 &p_vector2) const {
		return (x - p_vector2.x) * (x - p_vector2.x) + (y - p_vector2.y) * (y - p_vector2.y);
	}

	inline real_t angle_to(const Vector2 &p_vector2) const {
		return atan2(cross(p_vector2), dot(p_vector2));
	}

	inline real_t angle_to_point(const Vector2 &p_vector2) const {
		return atan2(y - p_vector2.y, x - p_vector2.x);
	}

	inline Vector2 direction_to(const Vector2 &p_b) const {
		Vector2 ret(p_b.x - x, p_b.y - y);
		ret.normalize();
		return ret;
	}

	inline real_t dot(const Vector2 &p_other) const {
		return x * p_other.x + y * p_other.y;
	}

	inline real_t cross(const Vector2 &p_other) const {
		return x * p_other.y - y * p_other.x;
	}

	inline Vector2 cross(real_t p_other) const {
		return Vector2(p_other * y, -p_other * x);
	}

	Vector2 project(const Vector2 &p_vec) const;

	Vector2 plane_project(real_t p_d, const Vector2 &p_vec) const;

	Vector2 clamped(real_t p_len) const;

	static inline Vector2 linear_interpolate(const Vector2 &p_a, const Vector2 &p_b, real_t p_t) {
		Vector2 res = p_a;
		res.x += (p_t * (p_b.x - p_a.x));
		res.y += (p_t * (p_b.y - p_a.y));
		return res;
	}

	inline Vector2 linear_interpolate(const Vector2 &p_b, real_t p_t) const {
		Vector2 res = *this;
		res.x += (p_t * (p_b.x - x));
		res.y += (p_t * (p_b.y - y));
		return res;
	}

	Vector2 cubic_interpolate(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, real_t p_t) const;

	Vector2 move_toward(const Vector2 &p_to, const real_t p_delta) const {
		Vector2 v = *this;
		Vector2 vd = p_to - v;
		real_t len = vd.length();
		return len <= p_delta || len < CMP_EPSILON ? p_to : v + vd / len * p_delta;
	}

	inline Vector2 slide(const Vector2 &p_vec) const {
		return p_vec - *this * this->dot(p_vec);
	}

	inline Vector2 bounce(const Vector2 &p_normal) const {
		return -reflect(p_normal);
	}

	inline Vector2 reflect(const Vector2 &p_normal) const {
		return -(*this - p_normal * this->dot(p_normal) * 2.0);
	}

	inline real_t angle() const {
		return atan2(y, x);
	}

	inline void set_rotation(real_t p_radians) {
		x = cosf(p_radians);
		y = sinf(p_radians);
	}

	inline Vector2 abs() const {
		return Vector2(fabs(x), fabs(y));
	}

	inline Vector2 rotated(real_t p_by) const {
		Vector2 v;
		v.set_rotation(angle() + p_by);
		v *= length();
		return v;
	}

	inline Vector2 tangent() const {
		return Vector2(y, -x);
	}

	inline Vector2 floor() const {
		return Vector2(Math::floor(x), Math::floor(y));
	}

	inline Vector2 snapped(const Vector2 &p_by) const {
		return Vector2(
				Math::stepify(x, p_by.x),
				Math::stepify(y, p_by.y));
	}

	inline real_t aspect() const { return width / height; }

	operator String() const;
};

inline Vector2 operator*(real_t p_scalar, const Vector2 &p_vec) {
	return p_vec * p_scalar;
}

namespace Math {

// Convenience, since they exist in GDScript

inline Vector2 cartesian2polar(Vector2 v) {
	return Vector2(Math::sqrt(v.x * v.x + v.y * v.y), Math::atan2(v.y, v.x));
}

inline Vector2 polar2cartesian(Vector2 v) {
	// x == radius
	// y == angle
	return Vector2(v.x * Math::cos(v.y), v.x * Math::sin(v.y));
}

} // namespace Math

} // namespace godot

#endif // VECTOR2_H

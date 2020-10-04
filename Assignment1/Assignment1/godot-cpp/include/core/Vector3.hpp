#ifndef VECTOR3_H
#define VECTOR3_H

#include <gdnative/vector3.h>

#include "Defs.hpp"

#include "String.hpp"

#include <Math.hpp>

namespace godot {

class Basis;

struct Vector3 {

	enum Axis {
		AXIS_X,
		AXIS_Y,
		AXIS_Z,
	};

	union {
		struct {
			real_t x;
			real_t y;
			real_t z;
		};

		real_t coord[3]; // Not for direct access, use [] operator instead
	};

	inline Vector3(real_t x, real_t y, real_t z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	inline Vector3() {
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}

	inline const real_t &operator[](int p_axis) const {
		return coord[p_axis];
	}

	inline real_t &operator[](int p_axis) {
		return coord[p_axis];
	}

	inline Vector3 &operator+=(const Vector3 &p_v) {
		x += p_v.x;
		y += p_v.y;
		z += p_v.z;
		return *this;
	}

	inline Vector3 operator+(const Vector3 &p_v) const {
		Vector3 v = *this;
		v += p_v;
		return v;
	}

	inline Vector3 &operator-=(const Vector3 &p_v) {
		x -= p_v.x;
		y -= p_v.y;
		z -= p_v.z;
		return *this;
	}

	inline Vector3 operator-(const Vector3 &p_v) const {
		Vector3 v = *this;
		v -= p_v;
		return v;
	}

	inline Vector3 &operator*=(const Vector3 &p_v) {
		x *= p_v.x;
		y *= p_v.y;
		z *= p_v.z;
		return *this;
	}

	inline Vector3 operator*(const Vector3 &p_v) const {
		Vector3 v = *this;
		v *= p_v;
		return v;
	}

	inline Vector3 &operator/=(const Vector3 &p_v) {
		x /= p_v.x;
		y /= p_v.y;
		z /= p_v.z;
		return *this;
	}

	inline Vector3 operator/(const Vector3 &p_v) const {
		Vector3 v = *this;
		v /= p_v;
		return v;
	}

	inline Vector3 &operator*=(real_t p_scalar) {
		*this *= Vector3(p_scalar, p_scalar, p_scalar);
		return *this;
	}

	inline Vector3 operator*(real_t p_scalar) const {
		Vector3 v = *this;
		v *= p_scalar;
		return v;
	}

	inline Vector3 &operator/=(real_t p_scalar) {
		*this /= Vector3(p_scalar, p_scalar, p_scalar);
		return *this;
	}

	inline Vector3 operator/(real_t p_scalar) const {
		Vector3 v = *this;
		v /= p_scalar;
		return v;
	}

	inline Vector3 operator-() const {
		return Vector3(-x, -y, -z);
	}

	inline bool operator==(const Vector3 &p_v) const {
		return (x == p_v.x && y == p_v.y && z == p_v.z);
	}

	inline bool operator!=(const Vector3 &p_v) const {
		return (x != p_v.x || y != p_v.y || z != p_v.z);
	}

	bool operator<(const Vector3 &p_v) const;

	bool operator<=(const Vector3 &p_v) const;

	inline Vector3 abs() const {
		return Vector3(::fabs(x), ::fabs(y), ::fabs(z));
	}

	inline Vector3 ceil() const {
		return Vector3(::ceil(x), ::ceil(y), ::ceil(z));
	}

	inline Vector3 cross(const Vector3 &b) const {
		Vector3 ret(
				(y * b.z) - (z * b.y),
				(z * b.x) - (x * b.z),
				(x * b.y) - (y * b.x));

		return ret;
	}

	inline Vector3 linear_interpolate(const Vector3 &p_b, real_t p_t) const {
		return Vector3(
				x + (p_t * (p_b.x - x)),
				y + (p_t * (p_b.y - y)),
				z + (p_t * (p_b.z - z)));
	}

	Vector3 cubic_interpolate(const Vector3 &b, const Vector3 &pre_a, const Vector3 &post_b, const real_t t) const;

	Vector3 move_toward(const Vector3 &p_to, const real_t p_delta) const {
		Vector3 v = *this;
		Vector3 vd = p_to - v;
		real_t len = vd.length();
		return len <= p_delta || len < CMP_EPSILON ? p_to : v + vd / len * p_delta;
	}

	Vector3 bounce(const Vector3 &p_normal) const {
		return -reflect(p_normal);
	}

	inline real_t length() const {
		real_t x2 = x * x;
		real_t y2 = y * y;
		real_t z2 = z * z;

		return ::sqrt(x2 + y2 + z2);
	}

	inline real_t length_squared() const {
		real_t x2 = x * x;
		real_t y2 = y * y;
		real_t z2 = z * z;

		return x2 + y2 + z2;
	}

	inline real_t distance_squared_to(const Vector3 &b) const {
		return (b - *this).length_squared();
	}

	inline real_t distance_to(const Vector3 &b) const {
		return (b - *this).length();
	}

	inline real_t dot(const Vector3 &b) const {
		return x * b.x + y * b.y + z * b.z;
	}

	inline real_t angle_to(const Vector3 &b) const {
		return std::atan2(cross(b).length(), dot(b));
	}

	inline Vector3 direction_to(const Vector3 &p_b) const {
		Vector3 ret(p_b.x - x, p_b.y - y, p_b.z - z);
		ret.normalize();
		return ret;
	}

	inline Vector3 floor() const {
		return Vector3(::floor(x), ::floor(y), ::floor(z));
	}

	inline Vector3 inverse() const {
		return Vector3(1.f / x, 1.f / y, 1.f / z);
	}

	inline bool is_normalized() const {
		return std::abs(length_squared() - 1.f) < 0.00001f;
	}

	Basis outer(const Vector3 &b) const;

	int max_axis() const;

	int min_axis() const;

	inline void normalize() {
		real_t l = length();
		if (l == 0) {
			x = y = z = 0;
		} else {
			x /= l;
			y /= l;
			z /= l;
		}
	}

	inline Vector3 normalized() const {
		Vector3 v = *this;
		v.normalize();
		return v;
	}

	inline Vector3 reflect(const Vector3 &p_normal) const {
		return -(*this - p_normal * this->dot(p_normal) * 2.0);
	}

	inline Vector3 rotated(const Vector3 &axis, const real_t phi) const {
		Vector3 v = *this;
		v.rotate(axis, phi);
		return v;
	}

	void rotate(const Vector3 &p_axis, real_t p_phi);

	inline Vector3 slide(const Vector3 &by) const {
		return by - *this * this->dot(by);
	}

	void snap(real_t p_val);

	inline Vector3 snapped(const float by) {
		Vector3 v = *this;
		v.snap(by);
		return v;
	}

	operator String() const;
};

inline Vector3 operator*(real_t p_scalar, const Vector3 &p_vec) {
	return p_vec * p_scalar;
}

inline Vector3 vec3_cross(const Vector3 &p_a, const Vector3 &p_b) {

	return p_a.cross(p_b);
}

} // namespace godot

#endif // VECTOR3_H

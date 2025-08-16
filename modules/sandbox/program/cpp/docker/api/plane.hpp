#pragma once
#include "vector.hpp"

struct Plane {
	Vector3 normal = Vector3(0, 1, 0);
	real_t d = 0.0f;

	void set_normal(const Vector3 &p_normal) { normal = p_normal; }
	const Vector3 &get_normal() const { return normal; }

	Vector3 center() const { return normal * d; }
	Vector3 get_any_perpendicular_normal() const;

	bool is_point_over(const Vector3 &p_point) const { return normal.dot(p_point) > d; }
	real_t distance_to(const Vector3 &p_point) const { return normal.dot(p_point) - d; }
	bool has_point(const Vector3 &p_point, real_t p_tolerance = 0.00001f) const;

	Vector3 project(const Vector3 &p_point) const {
		return p_point - normal * distance_to(p_point);
	}

	bool operator==(const Plane &p_plane) const;
	bool operator!=(const Plane &p_plane) const;

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	constexpr Plane() {}
	constexpr Plane(real_t p_a, real_t p_b, real_t p_c, real_t p_d) :
			normal(p_a, p_b, p_c),
			d(p_d) {}

	constexpr Plane(const Vector3 &p_normal, real_t p_d = 0.0);
	Plane(const Vector3 &p_normal, const Vector3 &p_point);
	Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir = CLOCKWISE);
};

inline constexpr Plane::Plane(const Vector3 &p_normal, real_t p_d) :
		normal(p_normal),
		d(p_d) {
}

inline Plane::Plane(const Vector3 &p_normal, const Vector3 &p_point) :
		normal(p_normal),
		d(p_normal.dot(p_point)) {
}

inline Plane::Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir) {
	if (p_dir == CLOCKWISE) {
		normal = (p_point1 - p_point3).cross(p_point1 - p_point2);
	} else {
		normal = (p_point1 - p_point2).cross(p_point1 - p_point3);
	}

	normal.normalize();
	d = normal.dot(p_point1);
}

inline bool Plane::has_point(const Vector3 &p_point, real_t p_tolerance) const {
	real_t dist = normal.dot(p_point) - d;
	dist = fabs(dist);
	return dist <= p_tolerance;
}

inline Vector3 Plane::get_any_perpendicular_normal() const {
	Vector3 n = normal;
	n.normalize();
	Vector3 tangent = n.cross(Vector3(1, 0, 0));
	if (tangent.length_squared() < 0.00001f) {
		tangent = n.cross(Vector3(0, 1, 0));
	}
	tangent.normalize();
	return n.cross(tangent);
}

inline bool Plane::operator==(const Plane &p_plane) const {
	return normal == p_plane.normal && d == p_plane.d;
}

inline bool Plane::operator!=(const Plane &p_plane) const {
	return normal != p_plane.normal || d != p_plane.d;
}

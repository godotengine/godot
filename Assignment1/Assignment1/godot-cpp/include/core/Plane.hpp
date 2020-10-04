#ifndef PLANE_H
#define PLANE_H

#include "Vector3.hpp"

#include <cmath>

namespace godot {

enum ClockDirection {

	CLOCKWISE,
	COUNTERCLOCKWISE
};

class Plane {
public:
	Vector3 normal;
	real_t d;

	void set_normal(const Vector3 &p_normal);

	inline Vector3 get_normal() const { return normal; } ///Point is coplanar, CMP_EPSILON for precision

	void normalize();

	Plane normalized() const;

	/* Plane-Point operations */

	inline Vector3 center() const { return normal * d; }
	Vector3 get_any_point() const;
	Vector3 get_any_perpendicular_normal() const;

	bool is_point_over(const Vector3 &p_point) const; ///< Point is over plane
	real_t distance_to(const Vector3 &p_point) const;
	bool has_point(const Vector3 &p_point, real_t _epsilon = CMP_EPSILON) const;

	/* intersections */

	bool intersect_3(const Plane &p_plane1, const Plane &p_plane2, Vector3 *r_result = 0) const;
	bool intersects_ray(Vector3 p_from, Vector3 p_dir, Vector3 *p_intersection) const;
	bool intersects_segment(Vector3 p_begin, Vector3 p_end, Vector3 *p_intersection) const;

	Vector3 project(const Vector3 &p_point) const;

	/* misc */

	inline Plane operator-() const { return Plane(-normal, -d); }
	bool is_almost_like(const Plane &p_plane) const;

	bool operator==(const Plane &p_plane) const;
	bool operator!=(const Plane &p_plane) const;
	operator String() const;

	inline Plane() { d = 0; }
	inline Plane(real_t p_a, real_t p_b, real_t p_c, real_t p_d) :
			normal(p_a, p_b, p_c),
			d(p_d) {}

	Plane(const Vector3 &p_normal, real_t p_d);
	Plane(const Vector3 &p_point, const Vector3 &p_normal);
	Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir = CLOCKWISE);
};

} // namespace godot

#endif // PLANE_H

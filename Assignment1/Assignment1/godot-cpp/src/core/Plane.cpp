#include "Plane.hpp"
#include "Vector3.hpp"

#include <cmath>

namespace godot {

void Plane::set_normal(const Vector3 &p_normal) {
	this->normal = p_normal;
}

Vector3 Plane::project(const Vector3 &p_point) const {

	return p_point - normal * distance_to(p_point);
}

void Plane::normalize() {

	real_t l = normal.length();
	if (l == 0) {
		*this = Plane(0, 0, 0, 0);
		return;
	}
	normal /= l;
	d /= l;
}

Plane Plane::normalized() const {

	Plane p = *this;
	p.normalize();
	return p;
}

Vector3 Plane::get_any_point() const {

	return get_normal() * d;
}

Vector3 Plane::get_any_perpendicular_normal() const {

	static const Vector3 p1 = Vector3(1, 0, 0);
	static const Vector3 p2 = Vector3(0, 1, 0);
	Vector3 p;

	if (::fabs(normal.dot(p1)) > 0.99) // if too similar to p1
		p = p2; // use p2
	else
		p = p1; // use p1

	p -= normal * normal.dot(p);
	p.normalize();

	return p;
}

/* intersections */

bool Plane::intersect_3(const Plane &p_plane1, const Plane &p_plane2, Vector3 *r_result) const {

	const Plane &p_plane0 = *this;
	Vector3 normal0 = p_plane0.normal;
	Vector3 normal1 = p_plane1.normal;
	Vector3 normal2 = p_plane2.normal;

	real_t denom = vec3_cross(normal0, normal1).dot(normal2);

	if (::fabs(denom) <= CMP_EPSILON)
		return false;

	if (r_result) {
		*r_result = ((vec3_cross(normal1, normal2) * p_plane0.d) +
							(vec3_cross(normal2, normal0) * p_plane1.d) +
							(vec3_cross(normal0, normal1) * p_plane2.d)) /
					denom;
	}

	return true;
}

bool Plane::intersects_ray(Vector3 p_from, Vector3 p_dir, Vector3 *p_intersection) const {

	Vector3 segment = p_dir;
	real_t den = normal.dot(segment);

	//printf("den is %i\n",den);
	if (::fabs(den) <= CMP_EPSILON) {

		return false;
	}

	real_t dist = (normal.dot(p_from) - d) / den;
	//printf("dist is %i\n",dist);

	if (dist > CMP_EPSILON) { //this is a ray, before the emiting pos (p_from) doesnt exist

		return false;
	}

	dist = -dist;
	*p_intersection = p_from + segment * dist;

	return true;
}

bool Plane::intersects_segment(Vector3 p_begin, Vector3 p_end, Vector3 *p_intersection) const {

	Vector3 segment = p_begin - p_end;
	real_t den = normal.dot(segment);

	//printf("den is %i\n",den);
	if (::fabs(den) <= CMP_EPSILON) {

		return false;
	}

	real_t dist = (normal.dot(p_begin) - d) / den;
	//printf("dist is %i\n",dist);

	if (dist < -CMP_EPSILON || dist > (1.0 + CMP_EPSILON)) {

		return false;
	}

	dist = -dist;
	*p_intersection = p_begin + segment * dist;

	return true;
}

/* misc */

bool Plane::is_almost_like(const Plane &p_plane) const {

	return (normal.dot(p_plane.normal) > _PLANE_EQ_DOT_EPSILON && ::fabs(d - p_plane.d) < _PLANE_EQ_D_EPSILON);
}

Plane::operator String() const {

	// return normal.operator String() + ", " + rtos(d);
	return String(); // @Todo
}

bool Plane::is_point_over(const Vector3 &p_point) const {

	return (normal.dot(p_point) > d);
}

real_t Plane::distance_to(const Vector3 &p_point) const {

	return (normal.dot(p_point) - d);
}

bool Plane::has_point(const Vector3 &p_point, real_t _epsilon) const {

	real_t dist = normal.dot(p_point) - d;
	dist = ::fabs(dist);
	return (dist <= _epsilon);
}

Plane::Plane(const Vector3 &p_normal, real_t p_d) {

	normal = p_normal;
	d = p_d;
}

Plane::Plane(const Vector3 &p_point, const Vector3 &p_normal) {

	normal = p_normal;
	d = p_normal.dot(p_point);
}

Plane::Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir) {

	if (p_dir == CLOCKWISE)
		normal = (p_point1 - p_point3).cross(p_point1 - p_point2);
	else
		normal = (p_point1 - p_point2).cross(p_point1 - p_point3);

	normal.normalize();
	d = normal.dot(p_point1);
}

bool Plane::operator==(const Plane &p_plane) const {

	return normal == p_plane.normal && d == p_plane.d;
}

bool Plane::operator!=(const Plane &p_plane) const {

	return normal != p_plane.normal || d != p_plane.d;
}

} // namespace godot

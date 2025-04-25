/**************************************************************************/
/*  plane.h                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/math/vector3.h"

class Variant;

struct [[nodiscard]] Plane {
	Vector3 normal;
	real_t d = 0;

	_FORCE_INLINE_ void set_normal(const Vector3 &p_normal);
	_FORCE_INLINE_ Vector3 get_normal() const { return normal; }

	_FORCE_INLINE_ void normalize();
	_FORCE_INLINE_ Plane normalized() const;

	/* Plane-Point operations */

	_FORCE_INLINE_ Vector3 get_center() const { return normal * d; }
	_FORCE_INLINE_ Vector3 get_any_perpendicular_normal() const;

	_FORCE_INLINE_ bool is_point_over(const Vector3 &p_point) const; ///< Point is over plane
	_FORCE_INLINE_ real_t distance_to(const Vector3 &p_point) const;
	_FORCE_INLINE_ bool has_point(const Vector3 &p_point, real_t p_tolerance = CMP_EPSILON) const;

	/* intersections */

	_FORCE_INLINE_ bool intersect_3(const Plane &p_plane1, const Plane &p_plane2, Vector3 *r_result = nullptr) const;
	_FORCE_INLINE_ bool intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const;
	_FORCE_INLINE_ bool intersects_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 *p_intersection) const;

	// For Variant bindings.
	Variant intersect_3_bind(const Plane &p_plane1, const Plane &p_plane2) const;
	Variant intersects_ray_bind(const Vector3 &p_from, const Vector3 &p_dir) const;
	Variant intersects_segment_bind(const Vector3 &p_begin, const Vector3 &p_end) const;

	_FORCE_INLINE_ Vector3 project(const Vector3 &p_point) const {
		return p_point - normal * distance_to(p_point);
	}

	/* misc */

	_FORCE_INLINE_ constexpr Plane operator-() const { return Plane(-normal, -d); }
	_FORCE_INLINE_ bool is_equal_approx(const Plane &p_plane) const;
	_FORCE_INLINE_ bool is_same(const Plane &p_plane) const;
	_FORCE_INLINE_ bool is_equal_approx_any_side(const Plane &p_plane) const;
	_FORCE_INLINE_ bool is_finite() const;

	_FORCE_INLINE_ constexpr bool operator==(const Plane &p_plane) const;
	_FORCE_INLINE_ constexpr bool operator!=(const Plane &p_plane) const;
	operator String() const;

	_FORCE_INLINE_ constexpr Plane() = default;
	_FORCE_INLINE_ constexpr Plane(real_t p_a, real_t p_b, real_t p_c, real_t p_d) :
			normal(p_a, p_b, p_c),
			d(p_d) {}

	_FORCE_INLINE_ constexpr Plane(const Vector3 &p_normal, real_t p_d = 0.0);
	_FORCE_INLINE_ Plane(const Vector3 &p_normal, const Vector3 &p_point);
	_FORCE_INLINE_ Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir = CLOCKWISE);
};

bool Plane::is_point_over(const Vector3 &p_point) const {
	return (normal.dot(p_point) > d);
}

real_t Plane::distance_to(const Vector3 &p_point) const {
	return (normal.dot(p_point) - d);
}

bool Plane::has_point(const Vector3 &p_point, real_t p_tolerance) const {
	real_t dist = normal.dot(p_point) - d;
	dist = Math::abs(dist);
	return (dist <= p_tolerance);
}

constexpr Plane::Plane(const Vector3 &p_normal, real_t p_d) :
		normal(p_normal),
		d(p_d) {
}

Plane::Plane(const Vector3 &p_normal, const Vector3 &p_point) :
		normal(p_normal),
		d(p_normal.dot(p_point)) {
}

Plane::Plane(const Vector3 &p_point1, const Vector3 &p_point2, const Vector3 &p_point3, ClockDirection p_dir) {
	if (p_dir == CLOCKWISE) {
		normal = (p_point1 - p_point3).cross(p_point1 - p_point2);
	} else {
		normal = (p_point1 - p_point2).cross(p_point1 - p_point3);
	}

	normal.normalize();
	d = normal.dot(p_point1);
}

constexpr bool Plane::operator==(const Plane &p_plane) const {
	return normal == p_plane.normal && d == p_plane.d;
}

constexpr bool Plane::operator!=(const Plane &p_plane) const {
	return normal != p_plane.normal || d != p_plane.d;
}

void Plane::set_normal(const Vector3 &p_normal) {
	normal = p_normal;
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

Vector3 Plane::get_any_perpendicular_normal() const {
	static const Vector3 p1 = Vector3(1, 0, 0);
	static const Vector3 p2 = Vector3(0, 1, 0);
	Vector3 p;

	if (Math::abs(normal.dot(p1)) > 0.99f) { // if too similar to p1
		p = p2; // use p2
	} else {
		p = p1; // use p1
	}

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

	if (Math::is_zero_approx(denom)) {
		return false;
	}

	if (r_result) {
		*r_result = ((vec3_cross(normal1, normal2) * p_plane0.d) +
							(vec3_cross(normal2, normal0) * p_plane1.d) +
							(vec3_cross(normal0, normal1) * p_plane2.d)) /
				denom;
	}

	return true;
}

bool Plane::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {
	Vector3 segment = p_dir;
	real_t den = normal.dot(segment);

	if (Math::is_zero_approx(den)) {
		return false;
	}

	real_t dist = (normal.dot(p_from) - d) / den;

	if (dist > (real_t)CMP_EPSILON) { //this is a ray, before the emitting pos (p_from) doesn't exist

		return false;
	}

	dist = -dist;
	*p_intersection = p_from + segment * dist;

	return true;
}

bool Plane::intersects_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 *p_intersection) const {
	Vector3 segment = p_begin - p_end;
	real_t den = normal.dot(segment);

	if (Math::is_zero_approx(den)) {
		return false;
	}

	real_t dist = (normal.dot(p_begin) - d) / den;

	if (dist < (real_t)-CMP_EPSILON || dist > (1.0f + (real_t)CMP_EPSILON)) {
		return false;
	}

	dist = -dist;
	*p_intersection = p_begin + segment * dist;

	return true;
}

/* misc */

bool Plane::is_equal_approx_any_side(const Plane &p_plane) const {
	return (normal.is_equal_approx(p_plane.normal) && Math::is_equal_approx(d, p_plane.d)) || (normal.is_equal_approx(-p_plane.normal) && Math::is_equal_approx(d, -p_plane.d));
}

bool Plane::is_equal_approx(const Plane &p_plane) const {
	return normal.is_equal_approx(p_plane.normal) && Math::is_equal_approx(d, p_plane.d);
}

bool Plane::is_same(const Plane &p_plane) const {
	return normal.is_same(p_plane.normal) && Math::is_same(d, p_plane.d);
}

bool Plane::is_finite() const {
	return normal.is_finite() && Math::is_finite(d);
}

template <>
struct is_zero_constructible<Plane> : std::true_type {};

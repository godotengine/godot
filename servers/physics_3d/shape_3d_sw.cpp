/*************************************************************************/
/*  shape_3d_sw.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "shape_3d_sw.h"

#include "core/io/image.h"
#include "core/math/convex_hull.h"
#include "core/math/geometry_3d.h"
#include "core/templates/sort_array.h"

// HeightMapShape3DSW is based on Bullet btHeightfieldTerrainShape.

/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#define _EDGE_IS_VALID_SUPPORT_THRESHOLD 0.0002
#define _FACE_IS_VALID_SUPPORT_THRESHOLD 0.9998

#define _CYLINDER_EDGE_IS_VALID_SUPPORT_THRESHOLD 0.002
#define _CYLINDER_FACE_IS_VALID_SUPPORT_THRESHOLD 0.999

void Shape3DSW::configure(const AABB &p_aabb) {
	aabb = p_aabb;
	configured = true;
	for (Map<ShapeOwner3DSW *, int>::Element *E = owners.front(); E; E = E->next()) {
		ShapeOwner3DSW *co = (ShapeOwner3DSW *)E->key();
		co->_shape_changed();
	}
}

Vector3 Shape3DSW::get_support(const Vector3 &p_normal) const {
	Vector3 res;
	int amnt;
	FeatureType type;
	get_supports(p_normal, 1, &res, amnt, type);
	return res;
}

void Shape3DSW::add_owner(ShapeOwner3DSW *p_owner) {
	Map<ShapeOwner3DSW *, int>::Element *E = owners.find(p_owner);
	if (E) {
		E->get()++;
	} else {
		owners[p_owner] = 1;
	}
}

void Shape3DSW::remove_owner(ShapeOwner3DSW *p_owner) {
	Map<ShapeOwner3DSW *, int>::Element *E = owners.find(p_owner);
	ERR_FAIL_COND(!E);
	E->get()--;
	if (E->get() == 0) {
		owners.erase(E);
	}
}

bool Shape3DSW::is_owner(ShapeOwner3DSW *p_owner) const {
	return owners.has(p_owner);
}

const Map<ShapeOwner3DSW *, int> &Shape3DSW::get_owners() const {
	return owners;
}

Shape3DSW::Shape3DSW() {
	custom_bias = 0;
	configured = false;
}

Shape3DSW::~Shape3DSW() {
	ERR_FAIL_COND(owners.size());
}

Plane PlaneShape3DSW::get_plane() const {
	return plane;
}

void PlaneShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	// gibberish, a plane is infinity
	r_min = -1e7;
	r_max = 1e7;
}

Vector3 PlaneShape3DSW::get_support(const Vector3 &p_normal) const {
	return p_normal * 1e15;
}

bool PlaneShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	bool inters = plane.intersects_segment(p_begin, p_end, &r_result);
	if (inters) {
		r_normal = plane.normal;
	}
	return inters;
}

bool PlaneShape3DSW::intersect_point(const Vector3 &p_point) const {
	return plane.distance_to(p_point) < 0;
}

Vector3 PlaneShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	if (plane.is_point_over(p_point)) {
		return plane.project(p_point);
	} else {
		return p_point;
	}
}

Vector3 PlaneShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	return Vector3(); //wtf
}

void PlaneShape3DSW::_setup(const Plane &p_plane) {
	plane = p_plane;
	configure(AABB(Vector3(-1e4, -1e4, -1e4), Vector3(1e4 * 2, 1e4 * 2, 1e4 * 2)));
}

void PlaneShape3DSW::set_data(const Variant &p_data) {
	_setup(p_data);
}

Variant PlaneShape3DSW::get_data() const {
	return plane;
}

PlaneShape3DSW::PlaneShape3DSW() {
}

//

real_t RayShape3DSW::get_length() const {
	return length;
}

bool RayShape3DSW::get_slips_on_slope() const {
	return slips_on_slope;
}

void RayShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	// don't think this will be even used
	r_min = 0;
	r_max = 1;
}

Vector3 RayShape3DSW::get_support(const Vector3 &p_normal) const {
	if (p_normal.z > 0) {
		return Vector3(0, 0, length);
	} else {
		return Vector3(0, 0, 0);
	}
}

void RayShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	if (Math::abs(p_normal.z) < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {
		r_amount = 2;
		r_type = FEATURE_EDGE;
		r_supports[0] = Vector3(0, 0, 0);
		r_supports[1] = Vector3(0, 0, length);
	} else if (p_normal.z > 0) {
		r_amount = 1;
		r_type = FEATURE_POINT;
		*r_supports = Vector3(0, 0, length);
	} else {
		r_amount = 1;
		r_type = FEATURE_POINT;
		*r_supports = Vector3(0, 0, 0);
	}
}

bool RayShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	return false; //simply not possible
}

bool RayShape3DSW::intersect_point(const Vector3 &p_point) const {
	return false; //simply not possible
}

Vector3 RayShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	Vector3 s[2] = {
		Vector3(0, 0, 0),
		Vector3(0, 0, length)
	};

	return Geometry3D::get_closest_point_to_segment(p_point, s);
}

Vector3 RayShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	return Vector3();
}

void RayShape3DSW::_setup(real_t p_length, bool p_slips_on_slope) {
	length = p_length;
	slips_on_slope = p_slips_on_slope;
	configure(AABB(Vector3(0, 0, 0), Vector3(0.1, 0.1, length)));
}

void RayShape3DSW::set_data(const Variant &p_data) {
	Dictionary d = p_data;
	_setup(d["length"], d["slips_on_slope"]);
}

Variant RayShape3DSW::get_data() const {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	return d;
}

RayShape3DSW::RayShape3DSW() {
	length = 1;
	slips_on_slope = false;
}

/********** SPHERE *************/

real_t SphereShape3DSW::get_radius() const {
	return radius;
}

void SphereShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	real_t d = p_normal.dot(p_transform.origin);

	// figure out scale at point
	Vector3 local_normal = p_transform.basis.xform_inv(p_normal);
	real_t scale = local_normal.length();

	r_min = d - (radius)*scale;
	r_max = d + (radius)*scale;
}

Vector3 SphereShape3DSW::get_support(const Vector3 &p_normal) const {
	return p_normal * radius;
}

void SphereShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	*r_supports = p_normal * radius;
	r_amount = 1;
	r_type = FEATURE_POINT;
}

bool SphereShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	return Geometry3D::segment_intersects_sphere(p_begin, p_end, Vector3(), radius, &r_result, &r_normal);
}

bool SphereShape3DSW::intersect_point(const Vector3 &p_point) const {
	return p_point.length() < radius;
}

Vector3 SphereShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	Vector3 p = p_point;
	real_t l = p.length();
	if (l < radius) {
		return p_point;
	}
	return (p / l) * radius;
}

Vector3 SphereShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	real_t s = 0.4 * p_mass * radius * radius;
	return Vector3(s, s, s);
}

void SphereShape3DSW::_setup(real_t p_radius) {
	radius = p_radius;
	configure(AABB(Vector3(-radius, -radius, -radius), Vector3(radius * 2.0, radius * 2.0, radius * 2.0)));
}

void SphereShape3DSW::set_data(const Variant &p_data) {
	_setup(p_data);
}

Variant SphereShape3DSW::get_data() const {
	return radius;
}

SphereShape3DSW::SphereShape3DSW() {
	radius = 0;
}

/********** BOX *************/

void BoxShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	// no matter the angle, the box is mirrored anyway
	Vector3 local_normal = p_transform.basis.xform_inv(p_normal);

	real_t length = local_normal.abs().dot(half_extents);
	real_t distance = p_normal.dot(p_transform.origin);

	r_min = distance - length;
	r_max = distance + length;
}

Vector3 BoxShape3DSW::get_support(const Vector3 &p_normal) const {
	Vector3 point(
			(p_normal.x < 0) ? -half_extents.x : half_extents.x,
			(p_normal.y < 0) ? -half_extents.y : half_extents.y,
			(p_normal.z < 0) ? -half_extents.z : half_extents.z);

	return point;
}

void BoxShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	static const int next[3] = { 1, 2, 0 };
	static const int next2[3] = { 2, 0, 1 };

	for (int i = 0; i < 3; i++) {
		Vector3 axis;
		axis[i] = 1.0;
		real_t dot = p_normal.dot(axis);
		if (Math::abs(dot) > _FACE_IS_VALID_SUPPORT_THRESHOLD) {
			//Vector3 axis_b;

			bool neg = dot < 0;
			r_amount = 4;
			r_type = FEATURE_FACE;

			Vector3 point;
			point[i] = half_extents[i];

			int i_n = next[i];
			int i_n2 = next2[i];

			static const real_t sign[4][2] = {
				{ -1.0, 1.0 },
				{ 1.0, 1.0 },
				{ 1.0, -1.0 },
				{ -1.0, -1.0 },
			};

			for (int j = 0; j < 4; j++) {
				point[i_n] = sign[j][0] * half_extents[i_n];
				point[i_n2] = sign[j][1] * half_extents[i_n2];
				r_supports[j] = neg ? -point : point;
			}

			if (neg) {
				SWAP(r_supports[1], r_supports[2]);
				SWAP(r_supports[0], r_supports[3]);
			}

			return;
		}

		r_amount = 0;
	}

	for (int i = 0; i < 3; i++) {
		Vector3 axis;
		axis[i] = 1.0;

		if (Math::abs(p_normal.dot(axis)) < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {
			r_amount = 2;
			r_type = FEATURE_EDGE;

			int i_n = next[i];
			int i_n2 = next2[i];

			Vector3 point = half_extents;

			if (p_normal[i_n] < 0) {
				point[i_n] = -point[i_n];
			}
			if (p_normal[i_n2] < 0) {
				point[i_n2] = -point[i_n2];
			}

			r_supports[0] = point;
			point[i] = -point[i];
			r_supports[1] = point;
			return;
		}
	}
	/* USE POINT */

	Vector3 point(
			(p_normal.x < 0) ? -half_extents.x : half_extents.x,
			(p_normal.y < 0) ? -half_extents.y : half_extents.y,
			(p_normal.z < 0) ? -half_extents.z : half_extents.z);

	r_amount = 1;
	r_type = FEATURE_POINT;
	r_supports[0] = point;
}

bool BoxShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	AABB aabb(-half_extents, half_extents * 2.0);

	return aabb.intersects_segment(p_begin, p_end, &r_result, &r_normal);
}

bool BoxShape3DSW::intersect_point(const Vector3 &p_point) const {
	return (Math::abs(p_point.x) < half_extents.x && Math::abs(p_point.y) < half_extents.y && Math::abs(p_point.z) < half_extents.z);
}

Vector3 BoxShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	int outside = 0;
	Vector3 min_point;

	for (int i = 0; i < 3; i++) {
		if (Math::abs(p_point[i]) > half_extents[i]) {
			outside++;
			if (outside == 1) {
				//use plane if only one side matches
				Vector3 n;
				n[i] = SGN(p_point[i]);

				Plane p(n, half_extents[i]);
				min_point = p.project(p_point);
			}
		}
	}

	if (!outside) {
		return p_point; //it's inside, don't do anything else
	}

	if (outside == 1) { //if only above one plane, this plane clearly wins
		return min_point;
	}

	//check segments
	real_t min_distance = 1e20;
	Vector3 closest_vertex = half_extents * p_point.sign();
	Vector3 s[2] = {
		closest_vertex,
		closest_vertex
	};

	for (int i = 0; i < 3; i++) {
		s[1] = closest_vertex;
		s[1][i] = -s[1][i]; //edge

		Vector3 closest_edge = Geometry3D::get_closest_point_to_segment(p_point, s);

		real_t d = p_point.distance_to(closest_edge);
		if (d < min_distance) {
			min_point = closest_edge;
			min_distance = d;
		}
	}

	return min_point;
}

Vector3 BoxShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	real_t lx = half_extents.x;
	real_t ly = half_extents.y;
	real_t lz = half_extents.z;

	return Vector3((p_mass / 3.0) * (ly * ly + lz * lz), (p_mass / 3.0) * (lx * lx + lz * lz), (p_mass / 3.0) * (lx * lx + ly * ly));
}

void BoxShape3DSW::_setup(const Vector3 &p_half_extents) {
	half_extents = p_half_extents.abs();

	configure(AABB(-half_extents, half_extents * 2));
}

void BoxShape3DSW::set_data(const Variant &p_data) {
	_setup(p_data);
}

Variant BoxShape3DSW::get_data() const {
	return half_extents;
}

BoxShape3DSW::BoxShape3DSW() {
}

/********** CAPSULE *************/

void CapsuleShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	Vector3 n = p_transform.basis.xform_inv(p_normal).normalized();
	real_t h = (n.y > 0) ? height : -height;

	n *= radius;
	n.y += h * 0.5;

	r_max = p_normal.dot(p_transform.xform(n));
	r_min = p_normal.dot(p_transform.xform(-n));
}

Vector3 CapsuleShape3DSW::get_support(const Vector3 &p_normal) const {
	Vector3 n = p_normal;

	real_t h = (n.y > 0) ? height : -height;

	n *= radius;
	n.y += h * 0.5;
	return n;
}

void CapsuleShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	Vector3 n = p_normal;

	real_t d = n.y;

	if (Math::abs(d) < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {
		// make it flat
		n.y = 0.0;
		n.normalize();
		n *= radius;

		r_amount = 2;
		r_type = FEATURE_EDGE;
		r_supports[0] = n;
		r_supports[0].y += height * 0.5;
		r_supports[1] = n;
		r_supports[1].y -= height * 0.5;

	} else {
		real_t h = (d > 0) ? height : -height;

		n *= radius;
		n.y += h * 0.5;
		r_amount = 1;
		r_type = FEATURE_POINT;
		*r_supports = n;
	}
}

bool CapsuleShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	Vector3 norm = (p_end - p_begin).normalized();
	real_t min_d = 1e20;

	Vector3 res, n;
	bool collision = false;

	Vector3 auxres, auxn;
	bool collided;

	// test against cylinder and spheres :-|

	collided = Geometry3D::segment_intersects_cylinder(p_begin, p_end, height, radius, &auxres, &auxn, 1);

	if (collided) {
		real_t d = norm.dot(auxres);
		if (d < min_d) {
			min_d = d;
			res = auxres;
			n = auxn;
			collision = true;
		}
	}

	collided = Geometry3D::segment_intersects_sphere(p_begin, p_end, Vector3(0, height * 0.5, 0), radius, &auxres, &auxn);

	if (collided) {
		real_t d = norm.dot(auxres);
		if (d < min_d) {
			min_d = d;
			res = auxres;
			n = auxn;
			collision = true;
		}
	}

	collided = Geometry3D::segment_intersects_sphere(p_begin, p_end, Vector3(0, height * -0.5, 0), radius, &auxres, &auxn);

	if (collided) {
		real_t d = norm.dot(auxres);

		if (d < min_d) {
			min_d = d;
			res = auxres;
			n = auxn;
			collision = true;
		}
	}

	if (collision) {
		r_result = res;
		r_normal = n;
	}
	return collision;
}

bool CapsuleShape3DSW::intersect_point(const Vector3 &p_point) const {
	if (Math::abs(p_point.y) < height * 0.5) {
		return Vector3(p_point.x, 0, p_point.z).length() < radius;
	} else {
		Vector3 p = p_point;
		p.y = Math::abs(p.y) - height * 0.5;
		return p.length() < radius;
	}
}

Vector3 CapsuleShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	Vector3 s[2] = {
		Vector3(0, -height * 0.5, 0),
		Vector3(0, height * 0.5, 0),
	};

	Vector3 p = Geometry3D::get_closest_point_to_segment(p_point, s);

	if (p.distance_to(p_point) < radius) {
		return p_point;
	}

	return p + (p_point - p).normalized() * radius;
}

Vector3 CapsuleShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	// use bad AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.y * extents.y));
}

void CapsuleShape3DSW::_setup(real_t p_height, real_t p_radius) {
	height = p_height;
	radius = p_radius;
	configure(AABB(Vector3(-radius, -height * 0.5 - radius, -radius), Vector3(radius * 2, height + radius * 2.0, radius * 2)));
}

void CapsuleShape3DSW::set_data(const Variant &p_data) {
	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("radius"));
	ERR_FAIL_COND(!d.has("height"));
	_setup(d["height"], d["radius"]);
}

Variant CapsuleShape3DSW::get_data() const {
	Dictionary d;
	d["radius"] = radius;
	d["height"] = height;
	return d;
}

CapsuleShape3DSW::CapsuleShape3DSW() {
	height = radius = 0;
}

/********** CYLINDER *************/

void CylinderShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	Vector3 cylinder_axis = p_transform.basis.get_axis(1).normalized();
	real_t axis_dot = cylinder_axis.dot(p_normal);

	Vector3 local_normal = p_transform.basis.xform_inv(p_normal);
	real_t scale = local_normal.length();
	real_t scaled_radius = radius * scale;
	real_t scaled_height = height * scale;

	real_t length;
	if (Math::abs(axis_dot) > 1.0) {
		length = scaled_height * 0.5;
	} else {
		length = Math::abs(axis_dot * scaled_height * 0.5) + scaled_radius * Math::sqrt(1.0 - axis_dot * axis_dot);
	}

	real_t distance = p_normal.dot(p_transform.origin);

	r_min = distance - length;
	r_max = distance + length;
}

Vector3 CylinderShape3DSW::get_support(const Vector3 &p_normal) const {
	Vector3 n = p_normal;
	real_t h = (n.y > 0) ? height : -height;
	real_t s = Math::sqrt(n.x * n.x + n.z * n.z);
	if (Math::is_zero_approx(s)) {
		n.x = radius;
		n.y = h * 0.5;
		n.z = 0.0;
	} else {
		real_t d = radius / s;
		n.x = n.x * d;
		n.y = h * 0.5;
		n.z = n.z * d;
	}

	return n;
}

void CylinderShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	real_t d = p_normal.y;
	if (Math::abs(d) > _CYLINDER_FACE_IS_VALID_SUPPORT_THRESHOLD) {
		real_t h = (d > 0) ? height : -height;

		Vector3 n = p_normal;
		n.x = 0.0;
		n.z = 0.0;
		n.y = h * 0.5;

		r_amount = 3;
		r_type = FEATURE_CIRCLE;
		r_supports[0] = n;
		r_supports[1] = n;
		r_supports[1].x += radius;
		r_supports[2] = n;
		r_supports[2].z += radius;
	} else if (Math::abs(d) < _CYLINDER_EDGE_IS_VALID_SUPPORT_THRESHOLD) {
		// make it flat
		Vector3 n = p_normal;
		n.y = 0.0;
		n.normalize();
		n *= radius;

		r_amount = 2;
		r_type = FEATURE_EDGE;
		r_supports[0] = n;
		r_supports[0].y += height * 0.5;
		r_supports[1] = n;
		r_supports[1].y -= height * 0.5;
	} else {
		r_amount = 1;
		r_type = FEATURE_POINT;
		r_supports[0] = get_support(p_normal);
		return;

		Vector3 n = p_normal;
		real_t h = n.y * Math::sqrt(0.25 * height * height + radius * radius);
		if (Math::abs(h) > 1.0) {
			// Top or bottom surface.
			n.y = (n.y > 0.0) ? height * 0.5 : -height * 0.5;
		} else {
			// Lateral surface.
			n.y = height * 0.5 * h;
		}

		real_t s = Math::sqrt(n.x * n.x + n.z * n.z);
		if (Math::is_zero_approx(s)) {
			n.x = 0.0;
			n.z = 0.0;
		} else {
			real_t scaled_radius = radius / s;
			n.x = n.x * scaled_radius;
			n.z = n.z * scaled_radius;
		}

		r_supports[0] = n;
	}
}

bool CylinderShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	return Geometry3D::segment_intersects_cylinder(p_begin, p_end, height, radius, &r_result, &r_normal, 1);
}

bool CylinderShape3DSW::intersect_point(const Vector3 &p_point) const {
	if (Math::abs(p_point.y) < height * 0.5) {
		return Vector3(p_point.x, 0, p_point.z).length() < radius;
	}
	return false;
}

Vector3 CylinderShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	if (Math::absf(p_point.y) > height * 0.5) {
		// Project point to top disk.
		real_t dir = p_point.y > 0.0 ? 1.0 : -1.0;
		Vector3 circle_pos(0.0, dir * height * 0.5, 0.0);
		Plane circle_plane(circle_pos, Vector3(0.0, dir, 0.0));
		Vector3 proj_point = circle_plane.project(p_point);

		// Clip position.
		Vector3 delta_point_1 = proj_point - circle_pos;
		real_t dist_point_1 = delta_point_1.length_squared();
		if (!Math::is_zero_approx(dist_point_1)) {
			dist_point_1 = Math::sqrt(dist_point_1);
			proj_point = circle_pos + delta_point_1 * MIN(dist_point_1, radius) / dist_point_1;
		}

		return proj_point;
	} else {
		Vector3 s[2] = {
			Vector3(0, -height * 0.5, 0),
			Vector3(0, height * 0.5, 0),
		};

		Vector3 p = Geometry3D::get_closest_point_to_segment(p_point, s);

		if (p.distance_to(p_point) < radius) {
			return p_point;
		}

		return p + (p_point - p).normalized() * radius;
	}
}

Vector3 CylinderShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	// use bad AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.y * extents.y));
}

void CylinderShape3DSW::_setup(real_t p_height, real_t p_radius) {
	height = p_height;
	radius = p_radius;
	configure(AABB(Vector3(-radius, -height * 0.5, -radius), Vector3(radius * 2.0, height, radius * 2.0)));
}

void CylinderShape3DSW::set_data(const Variant &p_data) {
	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("radius"));
	ERR_FAIL_COND(!d.has("height"));
	_setup(d["height"], d["radius"]);
}

Variant CylinderShape3DSW::get_data() const {
	Dictionary d;
	d["radius"] = radius;
	d["height"] = height;
	return d;
}

CylinderShape3DSW::CylinderShape3DSW() {
	height = radius = 0;
}

/********** CONVEX POLYGON *************/

void ConvexPolygonShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	int vertex_count = mesh.vertices.size();
	if (vertex_count == 0) {
		return;
	}

	const Vector3 *vrts = &mesh.vertices[0];

	for (int i = 0; i < vertex_count; i++) {
		real_t d = p_normal.dot(p_transform.xform(vrts[i]));

		if (i == 0 || d > r_max) {
			r_max = d;
		}
		if (i == 0 || d < r_min) {
			r_min = d;
		}
	}
}

Vector3 ConvexPolygonShape3DSW::get_support(const Vector3 &p_normal) const {
	Vector3 n = p_normal;

	int vert_support_idx = -1;
	real_t support_max = 0;

	int vertex_count = mesh.vertices.size();
	if (vertex_count == 0) {
		return Vector3();
	}

	const Vector3 *vrts = &mesh.vertices[0];

	for (int i = 0; i < vertex_count; i++) {
		real_t d = n.dot(vrts[i]);

		if (i == 0 || d > support_max) {
			support_max = d;
			vert_support_idx = i;
		}
	}

	return vrts[vert_support_idx];
}

void ConvexPolygonShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int ec = mesh.edges.size();

	const Vector3 *vertices = mesh.vertices.ptr();
	int vc = mesh.vertices.size();

	r_amount = 0;
	ERR_FAIL_COND_MSG(vc == 0, "Convex polygon shape has no vertices.");

	//find vertex first
	real_t max = 0;
	int vtx = 0;

	for (int i = 0; i < vc; i++) {
		real_t d = p_normal.dot(vertices[i]);

		if (i == 0 || d > max) {
			max = d;
			vtx = i;
		}
	}

	for (int i = 0; i < fc; i++) {
		if (faces[i].plane.normal.dot(p_normal) > _FACE_IS_VALID_SUPPORT_THRESHOLD) {
			int ic = faces[i].indices.size();
			const int *ind = faces[i].indices.ptr();

			bool valid = false;
			for (int j = 0; j < ic; j++) {
				if (ind[j] == vtx) {
					valid = true;
					break;
				}
			}

			if (!valid) {
				continue;
			}

			int m = MIN(p_max, ic);
			for (int j = 0; j < m; j++) {
				r_supports[j] = vertices[ind[j]];
			}
			r_amount = m;
			r_type = FEATURE_FACE;
			return;
		}
	}

	for (int i = 0; i < ec; i++) {
		real_t dot = (vertices[edges[i].a] - vertices[edges[i].b]).normalized().dot(p_normal);
		dot = ABS(dot);
		if (dot < _EDGE_IS_VALID_SUPPORT_THRESHOLD && (edges[i].a == vtx || edges[i].b == vtx)) {
			r_amount = 2;
			r_type = FEATURE_EDGE;
			r_supports[0] = vertices[edges[i].a];
			r_supports[1] = vertices[edges[i].b];
			return;
		}
	}

	r_supports[0] = vertices[vtx];
	r_amount = 1;
	r_type = FEATURE_POINT;
}

bool ConvexPolygonShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	const Vector3 *vertices = mesh.vertices.ptr();

	Vector3 n = p_end - p_begin;
	real_t min = 1e20;
	bool col = false;

	for (int i = 0; i < fc; i++) {
		if (faces[i].plane.normal.dot(n) > 0) {
			continue; //opposing face
		}

		int ic = faces[i].indices.size();
		const int *ind = faces[i].indices.ptr();

		for (int j = 1; j < ic - 1; j++) {
			Face3 f(vertices[ind[0]], vertices[ind[j]], vertices[ind[j + 1]]);
			Vector3 result;
			if (f.intersects_segment(p_begin, p_end, &result)) {
				real_t d = n.dot(result);
				if (d < min) {
					min = d;
					r_result = result;
					r_normal = faces[i].plane.normal;
					col = true;
				}

				break;
			}
		}
	}

	return col;
}

bool ConvexPolygonShape3DSW::intersect_point(const Vector3 &p_point) const {
	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	for (int i = 0; i < fc; i++) {
		if (faces[i].plane.distance_to(p_point) >= 0) {
			return false;
		}
	}

	return true;
}

Vector3 ConvexPolygonShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	const Geometry3D::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();
	const Vector3 *vertices = mesh.vertices.ptr();

	bool all_inside = true;
	for (int i = 0; i < fc; i++) {
		if (!faces[i].plane.is_point_over(p_point)) {
			continue;
		}

		all_inside = false;
		bool is_inside = true;
		int ic = faces[i].indices.size();
		const int *indices = faces[i].indices.ptr();

		for (int j = 0; j < ic; j++) {
			Vector3 a = vertices[indices[j]];
			Vector3 b = vertices[indices[(j + 1) % ic]];
			Vector3 n = (a - b).cross(faces[i].plane.normal).normalized();
			if (Plane(a, n).is_point_over(p_point)) {
				is_inside = false;
				break;
			}
		}

		if (is_inside) {
			return faces[i].plane.project(p_point);
		}
	}

	if (all_inside) {
		return p_point;
	}

	real_t min_distance = 1e20;
	Vector3 min_point;

	//check edges
	const Geometry3D::MeshData::Edge *edges = mesh.edges.ptr();
	int ec = mesh.edges.size();
	for (int i = 0; i < ec; i++) {
		Vector3 s[2] = {
			vertices[edges[i].a],
			vertices[edges[i].b]
		};

		Vector3 closest = Geometry3D::get_closest_point_to_segment(p_point, s);
		real_t d = closest.distance_to(p_point);
		if (d < min_distance) {
			min_distance = d;
			min_point = closest;
		}
	}

	return min_point;
}

Vector3 ConvexPolygonShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	// use bad AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.y * extents.y));
}

void ConvexPolygonShape3DSW::_setup(const Vector<Vector3> &p_vertices) {
	Error err = ConvexHullComputer::convex_hull(p_vertices, mesh);
	if (err != OK) {
		ERR_PRINT("Failed to build convex hull");
	}

	AABB _aabb;

	for (int i = 0; i < mesh.vertices.size(); i++) {
		if (i == 0) {
			_aabb.position = mesh.vertices[i];
		} else {
			_aabb.expand_to(mesh.vertices[i]);
		}
	}

	configure(_aabb);
}

void ConvexPolygonShape3DSW::set_data(const Variant &p_data) {
	_setup(p_data);
}

Variant ConvexPolygonShape3DSW::get_data() const {
	return mesh.vertices;
}

ConvexPolygonShape3DSW::ConvexPolygonShape3DSW() {
}

/********** FACE POLYGON *************/

void FaceShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	for (int i = 0; i < 3; i++) {
		Vector3 v = p_transform.xform(vertex[i]);
		real_t d = p_normal.dot(v);

		if (i == 0 || d > r_max) {
			r_max = d;
		}

		if (i == 0 || d < r_min) {
			r_min = d;
		}
	}
}

Vector3 FaceShape3DSW::get_support(const Vector3 &p_normal) const {
	int vert_support_idx = -1;
	real_t support_max = 0;

	for (int i = 0; i < 3; i++) {
		real_t d = p_normal.dot(vertex[i]);

		if (i == 0 || d > support_max) {
			support_max = d;
			vert_support_idx = i;
		}
	}

	return vertex[vert_support_idx];
}

void FaceShape3DSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount, FeatureType &r_type) const {
	Vector3 n = p_normal;

	/** TEST FACE AS SUPPORT **/
	if (Math::abs(normal.dot(n)) > _FACE_IS_VALID_SUPPORT_THRESHOLD) {
		r_amount = 3;
		r_type = FEATURE_FACE;
		for (int i = 0; i < 3; i++) {
			r_supports[i] = vertex[i];
		}
		return;
	}

	/** FIND SUPPORT VERTEX **/

	int vert_support_idx = -1;
	real_t support_max = 0;

	for (int i = 0; i < 3; i++) {
		real_t d = n.dot(vertex[i]);

		if (i == 0 || d > support_max) {
			support_max = d;
			vert_support_idx = i;
		}
	}

	/** TEST EDGES AS SUPPORT **/

	for (int i = 0; i < 3; i++) {
		int nx = (i + 1) % 3;
		if (i != vert_support_idx && nx != vert_support_idx) {
			continue;
		}

		// check if edge is valid as a support
		real_t dot = (vertex[i] - vertex[nx]).normalized().dot(n);
		dot = ABS(dot);
		if (dot < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {
			r_amount = 2;
			r_type = FEATURE_EDGE;
			r_supports[0] = vertex[i];
			r_supports[1] = vertex[nx];
			return;
		}
	}

	r_amount = 1;
	r_type = FEATURE_POINT;
	r_supports[0] = vertex[vert_support_idx];
}

bool FaceShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	bool c = Geometry3D::segment_intersects_triangle(p_begin, p_end, vertex[0], vertex[1], vertex[2], &r_result);
	if (c) {
		r_normal = Plane(vertex[0], vertex[1], vertex[2]).normal;
		if (r_normal.dot(p_end - p_begin) > 0) {
			if (backface_collision) {
				r_normal = -r_normal;
			} else {
				c = false;
			}
		}
	}

	return c;
}

bool FaceShape3DSW::intersect_point(const Vector3 &p_point) const {
	return false; //face is flat
}

Vector3 FaceShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	return Face3(vertex[0], vertex[1], vertex[2]).get_closest_point_to(p_point);
}

Vector3 FaceShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	return Vector3(); // Sorry, but i don't think anyone cares, FaceShape!
}

FaceShape3DSW::FaceShape3DSW() {
	configure(AABB());
}

Vector<Vector3> ConcavePolygonShape3DSW::get_faces() const {
	Vector<Vector3> rfaces;
	rfaces.resize(faces.size() * 3);

	for (int i = 0; i < faces.size(); i++) {
		Face f = faces.get(i);

		for (int j = 0; j < 3; j++) {
			rfaces.set(i * 3 + j, vertices.get(f.indices[j]));
		}
	}

	return rfaces;
}

void ConcavePolygonShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	int count = vertices.size();
	if (count == 0) {
		r_min = 0;
		r_max = 0;
		return;
	}
	const Vector3 *vptr = vertices.ptr();

	for (int i = 0; i < count; i++) {
		real_t d = p_normal.dot(p_transform.xform(vptr[i]));

		if (i == 0 || d > r_max) {
			r_max = d;
		}
		if (i == 0 || d < r_min) {
			r_min = d;
		}
	}
}

Vector3 ConcavePolygonShape3DSW::get_support(const Vector3 &p_normal) const {
	int count = vertices.size();
	if (count == 0) {
		return Vector3();
	}

	const Vector3 *vptr = vertices.ptr();

	Vector3 n = p_normal;

	int vert_support_idx = -1;
	real_t support_max = 0;

	for (int i = 0; i < count; i++) {
		real_t d = n.dot(vptr[i]);

		if (i == 0 || d > support_max) {
			support_max = d;
			vert_support_idx = i;
		}
	}

	return vptr[vert_support_idx];
}

void ConcavePolygonShape3DSW::_cull_segment(int p_idx, _SegmentCullParams *p_params) const {
	const BVH *bvh = &p_params->bvh[p_idx];

	/*
	if (p_params->dir.dot(bvh->aabb.get_support(-p_params->dir))>p_params->min_d)
		return; //test against whole AABB, which isn't very costly
	*/

	//printf("addr: %p\n",bvh);
	if (!bvh->aabb.intersects_segment(p_params->from, p_params->to)) {
		return;
	}

	if (bvh->face_index >= 0) {
		const Face *f = &p_params->faces[bvh->face_index];
		FaceShape3DSW *face = p_params->face;
		face->normal = f->normal;
		face->vertex[0] = p_params->vertices[f->indices[0]];
		face->vertex[1] = p_params->vertices[f->indices[1]];
		face->vertex[2] = p_params->vertices[f->indices[2]];

		Vector3 res;
		Vector3 normal;
		if (face->intersect_segment(p_params->from, p_params->to, res, normal)) {
			real_t d = p_params->dir.dot(res) - p_params->dir.dot(p_params->from);
			if ((d > 0) && (d < p_params->min_d)) {
				p_params->min_d = d;
				p_params->result = res;
				p_params->normal = normal;
				p_params->collisions++;
			}
		}
	} else {
		if (bvh->left >= 0) {
			_cull_segment(bvh->left, p_params);
		}
		if (bvh->right >= 0) {
			_cull_segment(bvh->right, p_params);
		}
	}
}

bool ConcavePolygonShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {
	if (faces.size() == 0) {
		return false;
	}

	// unlock data
	const Face *fr = faces.ptr();
	const Vector3 *vr = vertices.ptr();
	const BVH *br = bvh.ptr();

	FaceShape3DSW face;
	face.backface_collision = backface_collision;

	_SegmentCullParams params;
	params.from = p_begin;
	params.to = p_end;
	params.dir = (p_end - p_begin).normalized();

	params.faces = fr;
	params.vertices = vr;
	params.bvh = br;

	params.face = &face;

	// cull
	_cull_segment(0, &params);

	if (params.collisions > 0) {
		r_result = params.result;
		r_normal = params.normal;
		return true;
	} else {
		return false;
	}
}

bool ConcavePolygonShape3DSW::intersect_point(const Vector3 &p_point) const {
	return false; //face is flat
}

Vector3 ConcavePolygonShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	return Vector3();
}

void ConcavePolygonShape3DSW::_cull(int p_idx, _CullParams *p_params) const {
	const BVH *bvh = &p_params->bvh[p_idx];

	if (!p_params->aabb.intersects(bvh->aabb)) {
		return;
	}

	if (bvh->face_index >= 0) {
		const Face *f = &p_params->faces[bvh->face_index];
		FaceShape3DSW *face = p_params->face;
		face->normal = f->normal;
		face->vertex[0] = p_params->vertices[f->indices[0]];
		face->vertex[1] = p_params->vertices[f->indices[1]];
		face->vertex[2] = p_params->vertices[f->indices[2]];
		p_params->callback(p_params->userdata, face);

	} else {
		if (bvh->left >= 0) {
			_cull(bvh->left, p_params);
		}

		if (bvh->right >= 0) {
			_cull(bvh->right, p_params);
		}
	}
}

void ConcavePolygonShape3DSW::cull(const AABB &p_local_aabb, Callback p_callback, void *p_userdata) const {
	// make matrix local to concave
	if (faces.size() == 0) {
		return;
	}

	AABB local_aabb = p_local_aabb;

	// unlock data
	const Face *fr = faces.ptr();
	const Vector3 *vr = vertices.ptr();
	const BVH *br = bvh.ptr();

	FaceShape3DSW face; // use this to send in the callback
	face.backface_collision = backface_collision;

	_CullParams params;
	params.aabb = local_aabb;
	params.face = &face;
	params.faces = fr;
	params.vertices = vr;
	params.bvh = br;
	params.callback = p_callback;
	params.userdata = p_userdata;

	// cull
	_cull(0, &params);
}

Vector3 ConcavePolygonShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	// use bad AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.y * extents.y));
}

struct _VolumeSW_BVH_Element {
	AABB aabb;
	Vector3 center;
	int face_index;
};

struct _VolumeSW_BVH_CompareX {
	_FORCE_INLINE_ bool operator()(const _VolumeSW_BVH_Element &a, const _VolumeSW_BVH_Element &b) const {
		return a.center.x < b.center.x;
	}
};

struct _VolumeSW_BVH_CompareY {
	_FORCE_INLINE_ bool operator()(const _VolumeSW_BVH_Element &a, const _VolumeSW_BVH_Element &b) const {
		return a.center.y < b.center.y;
	}
};

struct _VolumeSW_BVH_CompareZ {
	_FORCE_INLINE_ bool operator()(const _VolumeSW_BVH_Element &a, const _VolumeSW_BVH_Element &b) const {
		return a.center.z < b.center.z;
	}
};

struct _VolumeSW_BVH {
	AABB aabb;
	_VolumeSW_BVH *left;
	_VolumeSW_BVH *right;

	int face_index;
};

_VolumeSW_BVH *_volume_sw_build_bvh(_VolumeSW_BVH_Element *p_elements, int p_size, int &count) {
	_VolumeSW_BVH *bvh = memnew(_VolumeSW_BVH);

	if (p_size == 1) {
		//leaf
		bvh->aabb = p_elements[0].aabb;
		bvh->left = nullptr;
		bvh->right = nullptr;
		bvh->face_index = p_elements->face_index;
		count++;
		return bvh;
	} else {
		bvh->face_index = -1;
	}

	AABB aabb;
	for (int i = 0; i < p_size; i++) {
		if (i == 0) {
			aabb = p_elements[i].aabb;
		} else {
			aabb.merge_with(p_elements[i].aabb);
		}
	}
	bvh->aabb = aabb;
	switch (aabb.get_longest_axis_index()) {
		case 0: {
			SortArray<_VolumeSW_BVH_Element, _VolumeSW_BVH_CompareX> sort_x;
			sort_x.sort(p_elements, p_size);

		} break;
		case 1: {
			SortArray<_VolumeSW_BVH_Element, _VolumeSW_BVH_CompareY> sort_y;
			sort_y.sort(p_elements, p_size);
		} break;
		case 2: {
			SortArray<_VolumeSW_BVH_Element, _VolumeSW_BVH_CompareZ> sort_z;
			sort_z.sort(p_elements, p_size);
		} break;
	}

	int split = p_size / 2;
	bvh->left = _volume_sw_build_bvh(p_elements, split, count);
	bvh->right = _volume_sw_build_bvh(&p_elements[split], p_size - split, count);

	//printf("branch at %p - %i: %i\n",bvh,count,bvh->face_index);
	count++;
	return bvh;
}

void ConcavePolygonShape3DSW::_fill_bvh(_VolumeSW_BVH *p_bvh_tree, BVH *p_bvh_array, int &p_idx) {
	int idx = p_idx;

	p_bvh_array[idx].aabb = p_bvh_tree->aabb;
	p_bvh_array[idx].face_index = p_bvh_tree->face_index;
	//printf("%p - %i: %i(%p)  -- %p:%p\n",%p_bvh_array[idx],p_idx,p_bvh_array[i]->face_index,&p_bvh_tree->face_index,p_bvh_tree->left,p_bvh_tree->right);

	if (p_bvh_tree->left) {
		p_bvh_array[idx].left = ++p_idx;
		_fill_bvh(p_bvh_tree->left, p_bvh_array, p_idx);

	} else {
		p_bvh_array[p_idx].left = -1;
	}

	if (p_bvh_tree->right) {
		p_bvh_array[idx].right = ++p_idx;
		_fill_bvh(p_bvh_tree->right, p_bvh_array, p_idx);

	} else {
		p_bvh_array[p_idx].right = -1;
	}

	memdelete(p_bvh_tree);
}

void ConcavePolygonShape3DSW::_setup(const Vector<Vector3> &p_faces, bool p_backface_collision) {
	int src_face_count = p_faces.size();
	if (src_face_count == 0) {
		configure(AABB());
		return;
	}
	ERR_FAIL_COND(src_face_count % 3);
	src_face_count /= 3;

	const Vector3 *facesr = p_faces.ptr();

	Vector<_VolumeSW_BVH_Element> bvh_array;
	bvh_array.resize(src_face_count);

	_VolumeSW_BVH_Element *bvh_arrayw = bvh_array.ptrw();

	faces.resize(src_face_count);
	Face *facesw = faces.ptrw();

	vertices.resize(src_face_count * 3);

	Vector3 *verticesw = vertices.ptrw();

	AABB _aabb;

	for (int i = 0; i < src_face_count; i++) {
		Face3 face(facesr[i * 3 + 0], facesr[i * 3 + 1], facesr[i * 3 + 2]);

		bvh_arrayw[i].aabb = face.get_aabb();
		bvh_arrayw[i].center = bvh_arrayw[i].aabb.position + bvh_arrayw[i].aabb.size * 0.5;
		bvh_arrayw[i].face_index = i;
		facesw[i].indices[0] = i * 3 + 0;
		facesw[i].indices[1] = i * 3 + 1;
		facesw[i].indices[2] = i * 3 + 2;
		facesw[i].normal = face.get_plane().normal;
		verticesw[i * 3 + 0] = face.vertex[0];
		verticesw[i * 3 + 1] = face.vertex[1];
		verticesw[i * 3 + 2] = face.vertex[2];
		if (i == 0) {
			_aabb = bvh_arrayw[i].aabb;
		} else {
			_aabb.merge_with(bvh_arrayw[i].aabb);
		}
	}

	int count = 0;
	_VolumeSW_BVH *bvh_tree = _volume_sw_build_bvh(bvh_arrayw, src_face_count, count);

	bvh.resize(count + 1);

	BVH *bvh_arrayw2 = bvh.ptrw();

	int idx = 0;
	_fill_bvh(bvh_tree, bvh_arrayw2, idx);

	backface_collision = p_backface_collision;

	configure(_aabb); // this type of shape has no margin
}

void ConcavePolygonShape3DSW::set_data(const Variant &p_data) {
	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("faces"));

	_setup(d["faces"], d["backface_collision"]);
}

Variant ConcavePolygonShape3DSW::get_data() const {
	Dictionary d;
	d["faces"] = get_faces();
	d["backface_collision"] = backface_collision;

	return d;
}

ConcavePolygonShape3DSW::ConcavePolygonShape3DSW() {
}

/* HEIGHT MAP SHAPE */

Vector<float> HeightMapShape3DSW::get_heights() const {
	return heights;
}

int HeightMapShape3DSW::get_width() const {
	return width;
}

int HeightMapShape3DSW::get_depth() const {
	return depth;
}

void HeightMapShape3DSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
	//not very useful, but not very used either
	p_transform.xform(get_aabb()).project_range_in_plane(Plane(p_normal, 0), r_min, r_max);
}

Vector3 HeightMapShape3DSW::get_support(const Vector3 &p_normal) const {
	//not very useful, but not very used either
	return get_aabb().get_support(p_normal);
}

struct _HeightmapSegmentCullParams {
	Vector3 from;
	Vector3 to;
	Vector3 dir;

	Vector3 result;
	Vector3 normal;

	const HeightMapShape3DSW *heightmap = nullptr;
	FaceShape3DSW *face = nullptr;
};

_FORCE_INLINE_ bool _heightmap_face_cull_segment(_HeightmapSegmentCullParams &p_params) {
	Vector3 res;
	Vector3 normal;
	if (p_params.face->intersect_segment(p_params.from, p_params.to, res, normal)) {
		p_params.result = res;
		p_params.normal = normal;
		return true;
	}

	return false;
}

_FORCE_INLINE_ bool _heightmap_cell_cull_segment(_HeightmapSegmentCullParams &p_params, int p_x, int p_z) {
	// First triangle.
	p_params.heightmap->_get_point(p_x, p_z, p_params.face->vertex[0]);
	p_params.heightmap->_get_point(p_x + 1, p_z, p_params.face->vertex[1]);
	p_params.heightmap->_get_point(p_x, p_z + 1, p_params.face->vertex[2]);
	p_params.face->normal = Plane(p_params.face->vertex[0], p_params.face->vertex[1], p_params.face->vertex[2]).normal;
	if (_heightmap_face_cull_segment(p_params)) {
		return true;
	}

	// Second triangle.
	p_params.face->vertex[0] = p_params.face->vertex[1];
	p_params.heightmap->_get_point(p_x + 1, p_z + 1, p_params.face->vertex[1]);
	p_params.face->normal = Plane(p_params.face->vertex[0], p_params.face->vertex[1], p_params.face->vertex[2]).normal;
	if (_heightmap_face_cull_segment(p_params)) {
		return true;
	}

	return false;
}

bool HeightMapShape3DSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_point, Vector3 &r_normal) const {
	if (heights.is_empty()) {
		return false;
	}

	Vector3 local_begin = p_begin + local_origin;
	Vector3 local_end = p_end + local_origin;

	FaceShape3DSW face;
	face.backface_collision = false;

	_HeightmapSegmentCullParams params;
	params.from = p_begin;
	params.to = p_end;
	params.dir = (p_end - p_begin).normalized();
	params.heightmap = this;
	params.face = &face;

	// Quantize the ray begin/end.
	int begin_x = floor(local_begin.x);
	int begin_z = floor(local_begin.z);
	int end_x = floor(local_end.x);
	int end_z = floor(local_end.z);

	if ((begin_x == end_x) && (begin_z == end_z)) {
		// Simple case for rays that don't traverse the grid horizontally.
		// Just perform a test on the given cell.
		int x = CLAMP(begin_x, 0, width - 2);
		int z = CLAMP(begin_z, 0, depth - 2);
		if (_heightmap_cell_cull_segment(params, x, z)) {
			r_point = params.result;
			r_normal = params.normal;
			return true;
		}
	} else {
		// Perform grid query from projected ray.
		Vector2 ray_dir_proj(local_end.x - local_begin.x, local_end.z - local_begin.z);
		real_t ray_dist_proj = ray_dir_proj.length();

		if (ray_dist_proj < CMP_EPSILON) {
			ray_dir_proj = Vector2();
		} else {
			ray_dir_proj /= ray_dist_proj;
		}

		const int x_step = (ray_dir_proj.x > CMP_EPSILON) ? 1 : ((ray_dir_proj.x < -CMP_EPSILON) ? -1 : 0);
		const int z_step = (ray_dir_proj.y > CMP_EPSILON) ? 1 : ((ray_dir_proj.y < -CMP_EPSILON) ? -1 : 0);

		const real_t infinite = 1e20;
		const real_t delta_x = (x_step != 0) ? 1.f / Math::abs(ray_dir_proj.x) : infinite;
		const real_t delta_z = (z_step != 0) ? 1.f / Math::abs(ray_dir_proj.y) : infinite;

		real_t cross_x; // At which value of `param` we will cross a x-axis lane?
		real_t cross_z; // At which value of `param` we will cross a z-axis lane?

		// X initialization.
		if (x_step != 0) {
			if (x_step == 1) {
				cross_x = (ceil(local_begin.x) - local_begin.x) * delta_x;
			} else {
				cross_x = (local_begin.x - floor(local_begin.x)) * delta_x;
			}
		} else {
			cross_x = infinite; // Will never cross on X.
		}

		// Z initialization.
		if (z_step != 0) {
			if (z_step == 1) {
				cross_z = (ceil(local_begin.z) - local_begin.z) * delta_z;
			} else {
				cross_z = (local_begin.z - floor(local_begin.z)) * delta_z;
			}
		} else {
			cross_z = infinite; // Will never cross on Z.
		}

		int x = floor(local_begin.x);
		int z = floor(local_begin.z);

		// Workaround cases where the ray starts at an integer position.
		if (Math::abs(cross_x) < CMP_EPSILON) {
			cross_x += delta_x;
			// If going backwards, we should ignore the position we would get by the above flooring,
			// because the ray is not heading in that direction.
			if (x_step == -1) {
				x -= 1;
			}
		}

		if (Math::abs(cross_z) < CMP_EPSILON) {
			cross_z += delta_z;
			if (z_step == -1) {
				z -= 1;
			}
		}

		// Start inside the grid.
		int x_start = CLAMP(x, 0, width - 2);
		int z_start = CLAMP(z, 0, depth - 2);

		// Adjust initial cross values.
		cross_x += delta_x * x_step * (x_start - x);
		cross_z += delta_z * z_step * (z_start - z);

		x = x_start;
		z = z_start;

		if (_heightmap_cell_cull_segment(params, x, z)) {
			r_point = params.result;
			r_normal = params.normal;
			return true;
		}

		real_t dist = 0.0;
		while (true) {
			if (cross_x < cross_z) {
				// X lane.
				x += x_step;
				// Assign before advancing the param,
				// to be in sync with the initialization step.
				dist = cross_x;
				cross_x += delta_x;
			} else {
				// Z lane.
				z += z_step;
				dist = cross_z;
				cross_z += delta_z;
			}

			// Stop when outside the grid.
			if ((x < 0) || (z < 0) || (x >= width - 1) || (z >= depth - 1)) {
				break;
			}

			if (_heightmap_cell_cull_segment(params, x, z)) {
				r_point = params.result;
				r_normal = params.normal;
				return true;
			}

			if (dist > ray_dist_proj) {
				break;
			}
		}
	}

	return false;
}

bool HeightMapShape3DSW::intersect_point(const Vector3 &p_point) const {
	return false;
}

Vector3 HeightMapShape3DSW::get_closest_point_to(const Vector3 &p_point) const {
	return Vector3();
}

void HeightMapShape3DSW::_get_cell(const Vector3 &p_point, int &r_x, int &r_y, int &r_z) const {
	const AABB &aabb = get_aabb();

	Vector3 pos_local = aabb.position + local_origin;

	Vector3 clamped_point(p_point);
	clamped_point.x = CLAMP(p_point.x, pos_local.x, pos_local.x + aabb.size.x);
	clamped_point.y = CLAMP(p_point.y, pos_local.y, pos_local.y + aabb.size.y);
	clamped_point.z = CLAMP(p_point.z, pos_local.z, pos_local.x + aabb.size.z);

	r_x = (clamped_point.x < 0.0) ? (clamped_point.x - 0.5) : (clamped_point.x + 0.5);
	r_y = (clamped_point.y < 0.0) ? (clamped_point.y - 0.5) : (clamped_point.y + 0.5);
	r_z = (clamped_point.z < 0.0) ? (clamped_point.z - 0.5) : (clamped_point.z + 0.5);
}

void HeightMapShape3DSW::cull(const AABB &p_local_aabb, Callback p_callback, void *p_userdata) const {
	if (heights.is_empty()) {
		return;
	}

	AABB local_aabb = p_local_aabb;
	local_aabb.position += local_origin;

	// Quantize the aabb, and adjust the start/end ranges.
	int aabb_min[3];
	int aabb_max[3];
	_get_cell(local_aabb.position, aabb_min[0], aabb_min[1], aabb_min[2]);
	_get_cell(local_aabb.position + local_aabb.size, aabb_max[0], aabb_max[1], aabb_max[2]);

	// Expand the min/max quantized values.
	// This is to catch the case where the input aabb falls between grid points.
	for (int i = 0; i < 3; ++i) {
		aabb_min[i]--;
		aabb_max[i]++;
	}

	int start_x = MAX(0, aabb_min[0]);
	int end_x = MIN(width - 1, aabb_max[0]);
	int start_z = MAX(0, aabb_min[2]);
	int end_z = MIN(depth - 1, aabb_max[2]);

	FaceShape3DSW face;
	face.backface_collision = true;

	for (int z = start_z; z < end_z; z++) {
		for (int x = start_x; x < end_x; x++) {
			// First triangle.
			_get_point(x, z, face.vertex[0]);
			_get_point(x + 1, z, face.vertex[1]);
			_get_point(x, z + 1, face.vertex[2]);
			face.normal = Plane(face.vertex[0], face.vertex[2], face.vertex[1]).normal;
			p_callback(p_userdata, &face);

			// Second triangle.
			face.vertex[0] = face.vertex[1];
			_get_point(x + 1, z + 1, face.vertex[1]);
			face.normal = Plane(face.vertex[0], face.vertex[2], face.vertex[1]).normal;
			p_callback(p_userdata, &face);
		}
	}
}

Vector3 HeightMapShape3DSW::get_moment_of_inertia(real_t p_mass) const {
	// use bad AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.y * extents.y));
}

void HeightMapShape3DSW::_setup(const Vector<float> &p_heights, int p_width, int p_depth, real_t p_min_height, real_t p_max_height) {
	heights = p_heights;
	width = p_width;
	depth = p_depth;

	// Initialize aabb.
	AABB aabb;
	aabb.position = Vector3(0.0, p_min_height, 0.0);
	aabb.size = Vector3(p_width - 1, p_max_height - p_min_height, p_depth - 1);

	// Initialize origin as the aabb center.
	local_origin = aabb.position + 0.5 * aabb.size;
	local_origin.y = 0.0;

	aabb.position -= local_origin;

	configure(aabb);
}

void HeightMapShape3DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);

	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("width"));
	ERR_FAIL_COND(!d.has("depth"));
	ERR_FAIL_COND(!d.has("heights"));

	int width = d["width"];
	int depth = d["depth"];

	ERR_FAIL_COND(width <= 0.0);
	ERR_FAIL_COND(depth <= 0.0);

	Variant heights_variant = d["heights"];
	Vector<float> heights_buffer;
	if (heights_variant.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
		// Ready-to-use heights can be passed.
		heights_buffer = heights_variant;
	} else if (heights_variant.get_type() == Variant::OBJECT) {
		// If an image is passed, we have to convert it.
		// This would be expensive to do with a script, so it's nice to have it here.
		Ref<Image> image = heights_variant;
		ERR_FAIL_COND(image.is_null());
		ERR_FAIL_COND(image->get_format() != Image::FORMAT_RF);

		PackedByteArray im_data = image->get_data();
		heights_buffer.resize(image->get_width() * image->get_height());

		float *w = heights_buffer.ptrw();
		float *rp = (float *)im_data.ptr();
		for (int i = 0; i < heights_buffer.size(); ++i) {
			w[i] = rp[i];
		}
	} else {
		ERR_FAIL_MSG("Expected PackedFloat32Array or float Image.");
	}

	// Compute min and max heights or use precomputed values.
	real_t min_height = 0.0;
	real_t max_height = 0.0;
	if (d.has("min_height") && d.has("max_height")) {
		min_height = d["min_height"];
		max_height = d["max_height"];
	} else {
		int heights_size = heights.size();
		for (int i = 0; i < heights_size; ++i) {
			float h = heights[i];
			if (h < min_height) {
				min_height = h;
			} else if (h > max_height) {
				max_height = h;
			}
		}
	}

	ERR_FAIL_COND(min_height > max_height);

	ERR_FAIL_COND(heights_buffer.size() != (width * depth));

	// If specified, min and max height will be used as precomputed values.
	_setup(heights_buffer, width, depth, min_height, max_height);
}

Variant HeightMapShape3DSW::get_data() const {
	Dictionary d;
	d["width"] = width;
	d["depth"] = depth;

	const AABB &aabb = get_aabb();
	d["min_height"] = aabb.position.y;
	d["max_height"] = aabb.position.y + aabb.size.y;

	d["heights"] = heights;

	return d;
}

HeightMapShape3DSW::HeightMapShape3DSW() {
}

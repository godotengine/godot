/*************************************************************************/
/*  shape_sw.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "shape_sw.h"

#include "geometry.h"
#include "quick_hull.h"
#include "sort.h"

#define _POINT_SNAP 0.001953125
#define _EDGE_IS_VALID_SUPPORT_THRESHOLD 0.0002
#define _FACE_IS_VALID_SUPPORT_THRESHOLD 0.9998

void ShapeSW::configure(const Rect3 &p_aabb) {
	aabb = p_aabb;
	configured = true;
	for (Map<ShapeOwnerSW *, int>::Element *E = owners.front(); E; E = E->next()) {
		ShapeOwnerSW *co = (ShapeOwnerSW *)E->key();
		co->_shape_changed();
	}
}

Vector3 ShapeSW::get_support(const Vector3 &p_normal) const {

	Vector3 res;
	int amnt;
	get_supports(p_normal, 1, &res, amnt);
	return res;
}

void ShapeSW::add_owner(ShapeOwnerSW *p_owner) {

	Map<ShapeOwnerSW *, int>::Element *E = owners.find(p_owner);
	if (E) {
		E->get()++;
	} else {
		owners[p_owner] = 1;
	}
}

void ShapeSW::remove_owner(ShapeOwnerSW *p_owner) {

	Map<ShapeOwnerSW *, int>::Element *E = owners.find(p_owner);
	ERR_FAIL_COND(!E);
	E->get()--;
	if (E->get() == 0) {
		owners.erase(E);
	}
}

bool ShapeSW::is_owner(ShapeOwnerSW *p_owner) const {

	return owners.has(p_owner);
}

const Map<ShapeOwnerSW *, int> &ShapeSW::get_owners() const {
	return owners;
}

ShapeSW::ShapeSW() {

	custom_bias = 0;
	configured = false;
}

ShapeSW::~ShapeSW() {

	ERR_FAIL_COND(owners.size());
}

Plane PlaneShapeSW::get_plane() const {

	return plane;
}

void PlaneShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	// gibberish, a plane is infinity
	r_min = -1e7;
	r_max = 1e7;
}

Vector3 PlaneShapeSW::get_support(const Vector3 &p_normal) const {

	return p_normal * 1e15;
}

bool PlaneShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	bool inters = plane.intersects_segment(p_begin, p_end, &r_result);
	if (inters)
		r_normal = plane.normal;
	return inters;
}

bool PlaneShapeSW::intersect_point(const Vector3 &p_point) const {

	return plane.distance_to(p_point) < 0;
}

Vector3 PlaneShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	if (plane.is_point_over(p_point)) {
		return plane.project(p_point);
	} else {
		return p_point;
	}
}

Vector3 PlaneShapeSW::get_moment_of_inertia(real_t p_mass) const {

	return Vector3(); //wtf
}

void PlaneShapeSW::_setup(const Plane &p_plane) {

	plane = p_plane;
	configure(Rect3(Vector3(-1e4, -1e4, -1e4), Vector3(1e4 * 2, 1e4 * 2, 1e4 * 2)));
}

void PlaneShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant PlaneShapeSW::get_data() const {

	return plane;
}

PlaneShapeSW::PlaneShapeSW() {
}

//

real_t RayShapeSW::get_length() const {

	return length;
}

void RayShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	// don't think this will be even used
	r_min = 0;
	r_max = 1;
}

Vector3 RayShapeSW::get_support(const Vector3 &p_normal) const {

	if (p_normal.z > 0)
		return Vector3(0, 0, length);
	else
		return Vector3(0, 0, 0);
}

void RayShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

	if (Math::abs(p_normal.z) < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {

		r_amount = 2;
		r_supports[0] = Vector3(0, 0, 0);
		r_supports[1] = Vector3(0, 0, length);
	} else if (p_normal.z > 0) {
		r_amount = 1;
		*r_supports = Vector3(0, 0, length);
	} else {
		r_amount = 1;
		*r_supports = Vector3(0, 0, 0);
	}
}

bool RayShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	return false; //simply not possible
}

bool RayShapeSW::intersect_point(const Vector3 &p_point) const {

	return false; //simply not possible
}

Vector3 RayShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	Vector3 s[2] = {
		Vector3(0, 0, 0),
		Vector3(0, 0, length)
	};

	return Geometry::get_closest_point_to_segment(p_point, s);
}

Vector3 RayShapeSW::get_moment_of_inertia(real_t p_mass) const {

	return Vector3();
}

void RayShapeSW::_setup(real_t p_length) {

	length = p_length;
	configure(Rect3(Vector3(0, 0, 0), Vector3(0.1, 0.1, length)));
}

void RayShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant RayShapeSW::get_data() const {

	return length;
}

RayShapeSW::RayShapeSW() {

	length = 1;
}

/********** SPHERE *************/

real_t SphereShapeSW::get_radius() const {

	return radius;
}

void SphereShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	real_t d = p_normal.dot(p_transform.origin);

	// figure out scale at point
	Vector3 local_normal = p_transform.basis.xform_inv(p_normal);
	real_t scale = local_normal.length();

	r_min = d - (radius)*scale;
	r_max = d + (radius)*scale;
}

Vector3 SphereShapeSW::get_support(const Vector3 &p_normal) const {

	return p_normal * radius;
}

void SphereShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

	*r_supports = p_normal * radius;
	r_amount = 1;
}

bool SphereShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	return Geometry::segment_intersects_sphere(p_begin, p_end, Vector3(), radius, &r_result, &r_normal);
}

bool SphereShapeSW::intersect_point(const Vector3 &p_point) const {

	return p_point.length() < radius;
}

Vector3 SphereShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	Vector3 p = p_point;
	float l = p.length();
	if (l < radius)
		return p_point;
	return (p / l) * radius;
}

Vector3 SphereShapeSW::get_moment_of_inertia(real_t p_mass) const {

	real_t s = 0.4 * p_mass * radius * radius;
	return Vector3(s, s, s);
}

void SphereShapeSW::_setup(real_t p_radius) {

	radius = p_radius;
	configure(Rect3(Vector3(-radius, -radius, -radius), Vector3(radius * 2.0, radius * 2.0, radius * 2.0)));
}

void SphereShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant SphereShapeSW::get_data() const {

	return radius;
}

SphereShapeSW::SphereShapeSW() {

	radius = 0;
}

/********** BOX *************/

void BoxShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	// no matter the angle, the box is mirrored anyway
	Vector3 local_normal = p_transform.basis.xform_inv(p_normal);

	real_t length = local_normal.abs().dot(half_extents);
	real_t distance = p_normal.dot(p_transform.origin);

	r_min = distance - length;
	r_max = distance + length;
}

Vector3 BoxShapeSW::get_support(const Vector3 &p_normal) const {

	Vector3 point(
			(p_normal.x < 0) ? -half_extents.x : half_extents.x,
			(p_normal.y < 0) ? -half_extents.y : half_extents.y,
			(p_normal.z < 0) ? -half_extents.z : half_extents.z);

	return point;
}

void BoxShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

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
	r_supports[0] = point;
}

bool BoxShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	Rect3 aabb(-half_extents, half_extents * 2.0);

	return aabb.intersects_segment(p_begin, p_end, &r_result, &r_normal);
}

bool BoxShapeSW::intersect_point(const Vector3 &p_point) const {

	return (Math::abs(p_point.x) < half_extents.x && Math::abs(p_point.y) < half_extents.y && Math::abs(p_point.z) < half_extents.z);
}

Vector3 BoxShapeSW::get_closest_point_to(const Vector3 &p_point) const {

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

	if (!outside)
		return p_point; //it's inside, don't do anything else

	if (outside == 1) //if only above one plane, this plane clearly wins
		return min_point;

	//check segments
	float min_distance = 1e20;
	Vector3 closest_vertex = half_extents * p_point.sign();
	Vector3 s[2] = {
		closest_vertex,
		closest_vertex
	};

	for (int i = 0; i < 3; i++) {

		s[1] = closest_vertex;
		s[1][i] = -s[1][i]; //edge

		Vector3 closest_edge = Geometry::get_closest_point_to_segment(p_point, s);

		float d = p_point.distance_to(closest_edge);
		if (d < min_distance) {
			min_point = closest_edge;
			min_distance = d;
		}
	}

	return min_point;
}

Vector3 BoxShapeSW::get_moment_of_inertia(real_t p_mass) const {

	real_t lx = half_extents.x;
	real_t ly = half_extents.y;
	real_t lz = half_extents.z;

	return Vector3((p_mass / 3.0) * (ly * ly + lz * lz), (p_mass / 3.0) * (lx * lx + lz * lz), (p_mass / 3.0) * (lx * lx + ly * ly));
}

void BoxShapeSW::_setup(const Vector3 &p_half_extents) {

	half_extents = p_half_extents.abs();

	configure(Rect3(-half_extents, half_extents * 2));
}

void BoxShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant BoxShapeSW::get_data() const {

	return half_extents;
}

BoxShapeSW::BoxShapeSW() {
}

/********** CAPSULE *************/

void CapsuleShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	Vector3 n = p_transform.basis.xform_inv(p_normal).normalized();
	real_t h = (n.z > 0) ? height : -height;

	n *= radius;
	n.z += h * 0.5;

	r_max = p_normal.dot(p_transform.xform(n));
	r_min = p_normal.dot(p_transform.xform(-n));
	return;

	n = p_transform.basis.xform(n);

	real_t distance = p_normal.dot(p_transform.origin);
	real_t length = Math::abs(p_normal.dot(n));
	r_min = distance - length;
	r_max = distance + length;

	ERR_FAIL_COND(r_max < r_min);
}

Vector3 CapsuleShapeSW::get_support(const Vector3 &p_normal) const {

	Vector3 n = p_normal;

	real_t h = (n.z > 0) ? height : -height;

	n *= radius;
	n.z += h * 0.5;
	return n;
}

void CapsuleShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

	Vector3 n = p_normal;

	real_t d = n.z;

	if (Math::abs(d) < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {

		// make it flat
		n.z = 0.0;
		n.normalize();
		n *= radius;

		r_amount = 2;
		r_supports[0] = n;
		r_supports[0].z += height * 0.5;
		r_supports[1] = n;
		r_supports[1].z -= height * 0.5;

	} else {

		real_t h = (d > 0) ? height : -height;

		n *= radius;
		n.z += h * 0.5;
		r_amount = 1;
		*r_supports = n;
	}
}

bool CapsuleShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	Vector3 norm = (p_end - p_begin).normalized();
	real_t min_d = 1e20;

	Vector3 res, n;
	bool collision = false;

	Vector3 auxres, auxn;
	bool collided;

	// test against cylinder and spheres :-|

	collided = Geometry::segment_intersects_cylinder(p_begin, p_end, height, radius, &auxres, &auxn);

	if (collided) {
		real_t d = norm.dot(auxres);
		if (d < min_d) {
			min_d = d;
			res = auxres;
			n = auxn;
			collision = true;
		}
	}

	collided = Geometry::segment_intersects_sphere(p_begin, p_end, Vector3(0, 0, height * 0.5), radius, &auxres, &auxn);

	if (collided) {
		real_t d = norm.dot(auxres);
		if (d < min_d) {
			min_d = d;
			res = auxres;
			n = auxn;
			collision = true;
		}
	}

	collided = Geometry::segment_intersects_sphere(p_begin, p_end, Vector3(0, 0, height * -0.5), radius, &auxres, &auxn);

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

bool CapsuleShapeSW::intersect_point(const Vector3 &p_point) const {

	if (Math::abs(p_point.z) < height * 0.5) {
		return Vector3(p_point.x, p_point.y, 0).length() < radius;
	} else {
		Vector3 p = p_point;
		p.z = Math::abs(p.z) - height * 0.5;
		return p.length() < radius;
	}
}

Vector3 CapsuleShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	Vector3 s[2] = {
		Vector3(0, 0, -height * 0.5),
		Vector3(0, 0, height * 0.5),
	};

	Vector3 p = Geometry::get_closest_point_to_segment(p_point, s);

	if (p.distance_to(p_point) < radius)
		return p_point;

	return p + (p_point - p).normalized() * radius;
}

Vector3 CapsuleShapeSW::get_moment_of_inertia(real_t p_mass) const {

	// use crappy AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.y * extents.y + extents.y * extents.y));
}

void CapsuleShapeSW::_setup(real_t p_height, real_t p_radius) {

	height = p_height;
	radius = p_radius;
	configure(Rect3(Vector3(-radius, -radius, -height * 0.5 - radius), Vector3(radius * 2, radius * 2, height + radius * 2.0)));
}

void CapsuleShapeSW::set_data(const Variant &p_data) {

	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("radius"));
	ERR_FAIL_COND(!d.has("height"));
	_setup(d["height"], d["radius"]);
}

Variant CapsuleShapeSW::get_data() const {

	Dictionary d;
	d["radius"] = radius;
	d["height"] = height;
	return d;
}

CapsuleShapeSW::CapsuleShapeSW() {

	height = radius = 0;
}

/********** CONVEX POLYGON *************/

void ConvexPolygonShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	int vertex_count = mesh.vertices.size();
	if (vertex_count == 0)
		return;

	const Vector3 *vrts = &mesh.vertices[0];

	for (int i = 0; i < vertex_count; i++) {

		real_t d = p_normal.dot(p_transform.xform(vrts[i]));

		if (i == 0 || d > r_max)
			r_max = d;
		if (i == 0 || d < r_min)
			r_min = d;
	}
}

Vector3 ConvexPolygonShapeSW::get_support(const Vector3 &p_normal) const {

	Vector3 n = p_normal;

	int vert_support_idx = -1;
	real_t support_max = 0;

	int vertex_count = mesh.vertices.size();
	if (vertex_count == 0)
		return Vector3();

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

void ConvexPolygonShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int ec = mesh.edges.size();

	const Vector3 *vertices = mesh.vertices.ptr();
	int vc = mesh.vertices.size();

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

			if (!valid)
				continue;

			int m = MIN(p_max, ic);
			for (int j = 0; j < m; j++) {

				r_supports[j] = vertices[ind[j]];
			}
			r_amount = m;
			return;
		}
	}

	for (int i = 0; i < ec; i++) {

		real_t dot = (vertices[edges[i].a] - vertices[edges[i].b]).normalized().dot(p_normal);
		dot = ABS(dot);
		if (dot < _EDGE_IS_VALID_SUPPORT_THRESHOLD && (edges[i].a == vtx || edges[i].b == vtx)) {

			r_amount = 2;
			r_supports[0] = vertices[edges[i].a];
			r_supports[1] = vertices[edges[i].b];
			return;
		}
	}

	r_supports[0] = vertices[vtx];
	r_amount = 1;
}

bool ConvexPolygonShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	const Vector3 *vertices = mesh.vertices.ptr();

	Vector3 n = p_end - p_begin;
	real_t min = 1e20;
	bool col = false;

	for (int i = 0; i < fc; i++) {

		if (faces[i].plane.normal.dot(n) > 0)
			continue; //opposing face

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

bool ConvexPolygonShapeSW::intersect_point(const Vector3 &p_point) const {

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();

	for (int i = 0; i < fc; i++) {

		if (faces[i].plane.distance_to(p_point) >= 0)
			return false;
	}

	return true;
}

Vector3 ConvexPolygonShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	const Geometry::MeshData::Face *faces = mesh.faces.ptr();
	int fc = mesh.faces.size();
	const Vector3 *vertices = mesh.vertices.ptr();

	bool all_inside = true;
	for (int i = 0; i < fc; i++) {

		if (!faces[i].plane.is_point_over(p_point))
			continue;

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

	float min_distance = 1e20;
	Vector3 min_point;

	//check edges
	const Geometry::MeshData::Edge *edges = mesh.edges.ptr();
	int ec = mesh.edges.size();
	for (int i = 0; i < ec; i++) {

		Vector3 s[2] = {
			vertices[edges[i].a],
			vertices[edges[i].b]
		};

		Vector3 closest = Geometry::get_closest_point_to_segment(p_point, s);
		float d = closest.distance_to(p_point);
		if (d < min_distance) {
			min_distance = d;
			min_point = closest;
		}
	}

	return min_point;
}

Vector3 ConvexPolygonShapeSW::get_moment_of_inertia(real_t p_mass) const {

	// use crappy AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.y * extents.y + extents.y * extents.y));
}

void ConvexPolygonShapeSW::_setup(const Vector<Vector3> &p_vertices) {

	Error err = QuickHull::build(p_vertices, mesh);
	if (err != OK)
		ERR_PRINT("Failed to build QuickHull");

	Rect3 _aabb;

	for (int i = 0; i < mesh.vertices.size(); i++) {

		if (i == 0)
			_aabb.position = mesh.vertices[i];
		else
			_aabb.expand_to(mesh.vertices[i]);
	}

	configure(_aabb);
}

void ConvexPolygonShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant ConvexPolygonShapeSW::get_data() const {

	return mesh.vertices;
}

ConvexPolygonShapeSW::ConvexPolygonShapeSW() {
}

/********** FACE POLYGON *************/

void FaceShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	for (int i = 0; i < 3; i++) {

		Vector3 v = p_transform.xform(vertex[i]);
		real_t d = p_normal.dot(v);

		if (i == 0 || d > r_max)
			r_max = d;

		if (i == 0 || d < r_min)
			r_min = d;
	}
}

Vector3 FaceShapeSW::get_support(const Vector3 &p_normal) const {

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

void FaceShapeSW::get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const {

	Vector3 n = p_normal;

	/** TEST FACE AS SUPPORT **/
	if (normal.dot(n) > _FACE_IS_VALID_SUPPORT_THRESHOLD) {

		r_amount = 3;
		for (int i = 0; i < 3; i++) {

			r_supports[i] = vertex[i];
		}
		return;
	}

	/** FIND SUPPORT VERTEX **/

	int vert_support_idx = -1;
	real_t support_max;

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
		if (i != vert_support_idx && nx != vert_support_idx)
			continue;

		// check if edge is valid as a support
		real_t dot = (vertex[i] - vertex[nx]).normalized().dot(n);
		dot = ABS(dot);
		if (dot < _EDGE_IS_VALID_SUPPORT_THRESHOLD) {

			r_amount = 2;
			r_supports[0] = vertex[i];
			r_supports[1] = vertex[nx];
			return;
		}
	}

	r_amount = 1;
	r_supports[0] = vertex[vert_support_idx];
}

bool FaceShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	bool c = Geometry::segment_intersects_triangle(p_begin, p_end, vertex[0], vertex[1], vertex[2], &r_result);
	if (c) {
		r_normal = Plane(vertex[0], vertex[1], vertex[2]).normal;
		if (r_normal.dot(p_end - p_begin) > 0) {
			r_normal = -r_normal;
		}
	}

	return c;
}

bool FaceShapeSW::intersect_point(const Vector3 &p_point) const {

	return false; //face is flat
}

Vector3 FaceShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	return Face3(vertex[0], vertex[1], vertex[2]).get_closest_point_to(p_point);
}

Vector3 FaceShapeSW::get_moment_of_inertia(real_t p_mass) const {

	return Vector3(); // Sorry, but i don't think anyone cares, FaceShape!
}

FaceShapeSW::FaceShapeSW() {

	configure(Rect3());
}

PoolVector<Vector3> ConcavePolygonShapeSW::get_faces() const {

	PoolVector<Vector3> rfaces;
	rfaces.resize(faces.size() * 3);

	for (int i = 0; i < faces.size(); i++) {

		Face f = faces.get(i);

		for (int j = 0; j < 3; j++) {

			rfaces.set(i * 3 + j, vertices.get(f.indices[j]));
		}
	}

	return rfaces;
}

void ConcavePolygonShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	int count = vertices.size();
	if (count == 0) {
		r_min = 0;
		r_max = 0;
		return;
	}
	PoolVector<Vector3>::Read r = vertices.read();
	const Vector3 *vptr = r.ptr();

	for (int i = 0; i < count; i++) {

		real_t d = p_normal.dot(p_transform.xform(vptr[i]));

		if (i == 0 || d > r_max)
			r_max = d;
		if (i == 0 || d < r_min)
			r_min = d;
	}
}

Vector3 ConcavePolygonShapeSW::get_support(const Vector3 &p_normal) const {

	int count = vertices.size();
	if (count == 0)
		return Vector3();

	PoolVector<Vector3>::Read r = vertices.read();
	const Vector3 *vptr = r.ptr();

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

void ConcavePolygonShapeSW::_cull_segment(int p_idx, _SegmentCullParams *p_params) const {

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

		Vector3 res;
		Vector3 vertices[3] = {
			p_params->vertices[p_params->faces[bvh->face_index].indices[0]],
			p_params->vertices[p_params->faces[bvh->face_index].indices[1]],
			p_params->vertices[p_params->faces[bvh->face_index].indices[2]]
		};

		if (Geometry::segment_intersects_triangle(
					p_params->from,
					p_params->to,
					vertices[0],
					vertices[1],
					vertices[2],
					&res)) {

			real_t d = p_params->dir.dot(res) - p_params->dir.dot(p_params->from);
			//TODO, seems segmen/triangle intersection is broken :(
			if (d > 0 && d < p_params->min_d) {

				p_params->min_d = d;
				p_params->result = res;
				p_params->normal = Plane(vertices[0], vertices[1], vertices[2]).normal;
				if (p_params->normal.dot(p_params->dir) > 0)
					p_params->normal = -p_params->normal;
				p_params->collisions++;
			}
		}

	} else {

		if (bvh->left >= 0)
			_cull_segment(bvh->left, p_params);
		if (bvh->right >= 0)
			_cull_segment(bvh->right, p_params);
	}
}

bool ConcavePolygonShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const {

	if (faces.size() == 0)
		return false;

	// unlock data
	PoolVector<Face>::Read fr = faces.read();
	PoolVector<Vector3>::Read vr = vertices.read();
	PoolVector<BVH>::Read br = bvh.read();

	_SegmentCullParams params;
	params.from = p_begin;
	params.to = p_end;
	params.collisions = 0;
	params.dir = (p_end - p_begin).normalized();

	params.faces = fr.ptr();
	params.vertices = vr.ptr();
	params.bvh = br.ptr();

	params.min_d = 1e20;
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

bool ConcavePolygonShapeSW::intersect_point(const Vector3 &p_point) const {

	return false; //face is flat
}

Vector3 ConcavePolygonShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	return Vector3();
}

void ConcavePolygonShapeSW::_cull(int p_idx, _CullParams *p_params) const {

	const BVH *bvh = &p_params->bvh[p_idx];

	if (!p_params->aabb.intersects(bvh->aabb))
		return;

	if (bvh->face_index >= 0) {

		const Face *f = &p_params->faces[bvh->face_index];
		FaceShapeSW *face = p_params->face;
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

void ConcavePolygonShapeSW::cull(const Rect3 &p_local_aabb, Callback p_callback, void *p_userdata) const {

	// make matrix local to concave
	if (faces.size() == 0)
		return;

	Rect3 local_aabb = p_local_aabb;

	// unlock data
	PoolVector<Face>::Read fr = faces.read();
	PoolVector<Vector3>::Read vr = vertices.read();
	PoolVector<BVH>::Read br = bvh.read();

	FaceShapeSW face; // use this to send in the callback

	_CullParams params;
	params.aabb = local_aabb;
	params.face = &face;
	params.faces = fr.ptr();
	params.vertices = vr.ptr();
	params.bvh = br.ptr();
	params.callback = p_callback;
	params.userdata = p_userdata;

	// cull
	_cull(0, &params);
}

Vector3 ConcavePolygonShapeSW::get_moment_of_inertia(real_t p_mass) const {

	// use crappy AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.y * extents.y + extents.y * extents.y));
}

struct _VolumeSW_BVH_Element {

	Rect3 aabb;
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

	Rect3 aabb;
	_VolumeSW_BVH *left;
	_VolumeSW_BVH *right;

	int face_index;
};

_VolumeSW_BVH *_volume_sw_build_bvh(_VolumeSW_BVH_Element *p_elements, int p_size, int &count) {

	_VolumeSW_BVH *bvh = memnew(_VolumeSW_BVH);

	if (p_size == 1) {
		//leaf
		bvh->aabb = p_elements[0].aabb;
		bvh->left = NULL;
		bvh->right = NULL;
		bvh->face_index = p_elements->face_index;
		count++;
		return bvh;
	} else {

		bvh->face_index = -1;
	}

	Rect3 aabb;
	for (int i = 0; i < p_size; i++) {

		if (i == 0)
			aabb = p_elements[i].aabb;
		else
			aabb.merge_with(p_elements[i].aabb);
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

void ConcavePolygonShapeSW::_fill_bvh(_VolumeSW_BVH *p_bvh_tree, BVH *p_bvh_array, int &p_idx) {

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

void ConcavePolygonShapeSW::_setup(PoolVector<Vector3> p_faces) {

	int src_face_count = p_faces.size();
	if (src_face_count == 0) {
		configure(Rect3());
		return;
	}
	ERR_FAIL_COND(src_face_count % 3);
	src_face_count /= 3;

	PoolVector<Vector3>::Read r = p_faces.read();
	const Vector3 *facesr = r.ptr();

	PoolVector<_VolumeSW_BVH_Element> bvh_array;
	bvh_array.resize(src_face_count);

	PoolVector<_VolumeSW_BVH_Element>::Write bvhw = bvh_array.write();
	_VolumeSW_BVH_Element *bvh_arrayw = bvhw.ptr();

	faces.resize(src_face_count);
	PoolVector<Face>::Write w = faces.write();
	Face *facesw = w.ptr();

	vertices.resize(src_face_count * 3);

	PoolVector<Vector3>::Write vw = vertices.write();
	Vector3 *verticesw = vw.ptr();

	Rect3 _aabb;

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
		if (i == 0)
			_aabb = bvh_arrayw[i].aabb;
		else
			_aabb.merge_with(bvh_arrayw[i].aabb);
	}

	w = PoolVector<Face>::Write();
	vw = PoolVector<Vector3>::Write();

	int count = 0;
	_VolumeSW_BVH *bvh_tree = _volume_sw_build_bvh(bvh_arrayw, src_face_count, count);

	bvh.resize(count + 1);

	PoolVector<BVH>::Write bvhw2 = bvh.write();
	BVH *bvh_arrayw2 = bvhw2.ptr();

	int idx = 0;
	_fill_bvh(bvh_tree, bvh_arrayw2, idx);

	configure(_aabb); // this type of shape has no margin
}

void ConcavePolygonShapeSW::set_data(const Variant &p_data) {

	_setup(p_data);
}

Variant ConcavePolygonShapeSW::get_data() const {

	return get_faces();
}

ConcavePolygonShapeSW::ConcavePolygonShapeSW() {
}

/* HEIGHT MAP SHAPE */

PoolVector<real_t> HeightMapShapeSW::get_heights() const {

	return heights;
}
int HeightMapShapeSW::get_width() const {

	return width;
}
int HeightMapShapeSW::get_depth() const {

	return depth;
}
real_t HeightMapShapeSW::get_cell_size() const {

	return cell_size;
}

void HeightMapShapeSW::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	//not very useful, but not very used either
	p_transform.xform(get_aabb()).project_range_in_plane(Plane(p_normal, 0), r_min, r_max);
}

Vector3 HeightMapShapeSW::get_support(const Vector3 &p_normal) const {

	//not very useful, but not very used either
	return get_aabb().get_support(p_normal);
}

bool HeightMapShapeSW::intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_point, Vector3 &r_normal) const {

	return false;
}

bool HeightMapShapeSW::intersect_point(const Vector3 &p_point) const {
	return false;
}

Vector3 HeightMapShapeSW::get_closest_point_to(const Vector3 &p_point) const {

	return Vector3();
}

void HeightMapShapeSW::cull(const Rect3 &p_local_aabb, Callback p_callback, void *p_userdata) const {
}

Vector3 HeightMapShapeSW::get_moment_of_inertia(real_t p_mass) const {

	// use crappy AABB approximation
	Vector3 extents = get_aabb().size * 0.5;

	return Vector3(
			(p_mass / 3.0) * (extents.y * extents.y + extents.z * extents.z),
			(p_mass / 3.0) * (extents.x * extents.x + extents.z * extents.z),
			(p_mass / 3.0) * (extents.y * extents.y + extents.y * extents.y));
}

void HeightMapShapeSW::_setup(PoolVector<real_t> p_heights, int p_width, int p_depth, real_t p_cell_size) {

	heights = p_heights;
	width = p_width;
	depth = p_depth;
	cell_size = p_cell_size;

	PoolVector<real_t>::Read r = heights.read();

	Rect3 aabb;

	for (int i = 0; i < depth; i++) {

		for (int j = 0; j < width; j++) {

			real_t h = r[i * width + j];

			Vector3 pos(j * cell_size, h, i * cell_size);
			if (i == 0 || j == 0)
				aabb.position = pos;
			else
				aabb.expand_to(pos);
		}
	}

	configure(aabb);
}

void HeightMapShapeSW::set_data(const Variant &p_data) {

	ERR_FAIL_COND(p_data.get_type() != Variant::DICTIONARY);
	Dictionary d = p_data;
	ERR_FAIL_COND(!d.has("width"));
	ERR_FAIL_COND(!d.has("depth"));
	ERR_FAIL_COND(!d.has("cell_size"));
	ERR_FAIL_COND(!d.has("heights"));

	int width = d["width"];
	int depth = d["depth"];
	real_t cell_size = d["cell_size"];
	PoolVector<real_t> heights = d["heights"];

	ERR_FAIL_COND(width <= 0);
	ERR_FAIL_COND(depth <= 0);
	ERR_FAIL_COND(cell_size <= CMP_EPSILON);
	ERR_FAIL_COND(heights.size() != (width * depth));
	_setup(heights, width, depth, cell_size);
}

Variant HeightMapShapeSW::get_data() const {

	ERR_FAIL_V(Variant());
}

HeightMapShapeSW::HeightMapShapeSW() {

	width = 0;
	depth = 0;
	cell_size = 0;
}

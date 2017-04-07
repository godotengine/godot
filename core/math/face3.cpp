/*************************************************************************/
/*  face3.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "face3.h"
#include "geometry.h"

int Face3::split_by_plane(const Plane &p_plane, Face3 p_res[3], bool p_is_point_over[3]) const {

	ERR_FAIL_COND_V(is_degenerate(), 0);

	Vector3 above[4];
	int above_count = 0;

	Vector3 below[4];
	int below_count = 0;

	for (int i = 0; i < 3; i++) {

		if (p_plane.has_point(vertex[i], CMP_EPSILON)) { // point is in plane

			ERR_FAIL_COND_V(above_count >= 4, 0);
			above[above_count++] = vertex[i];
			ERR_FAIL_COND_V(below_count >= 4, 0);
			below[below_count++] = vertex[i];

		} else {

			if (p_plane.is_point_over(vertex[i])) {
				//Point is over
				ERR_FAIL_COND_V(above_count >= 4, 0);
				above[above_count++] = vertex[i];

			} else {
				//Point is under
				ERR_FAIL_COND_V(below_count >= 4, 0);
				below[below_count++] = vertex[i];
			}

			/* Check for Intersection between this and the next vertex*/

			Vector3 inters;
			if (!p_plane.intersects_segment(vertex[i], vertex[(i + 1) % 3], &inters))
				continue;

			/* Intersection goes to both */
			ERR_FAIL_COND_V(above_count >= 4, 0);
			above[above_count++] = inters;
			ERR_FAIL_COND_V(below_count >= 4, 0);
			below[below_count++] = inters;
		}
	}

	int polygons_created = 0;

	ERR_FAIL_COND_V(above_count >= 4 && below_count >= 4, 0); //bug in the algo

	if (above_count >= 3) {

		p_res[polygons_created] = Face3(above[0], above[1], above[2]);
		p_is_point_over[polygons_created] = true;
		polygons_created++;

		if (above_count == 4) {

			p_res[polygons_created] = Face3(above[2], above[3], above[0]);
			p_is_point_over[polygons_created] = true;
			polygons_created++;
		}
	}

	if (below_count >= 3) {

		p_res[polygons_created] = Face3(below[0], below[1], below[2]);
		p_is_point_over[polygons_created] = false;
		polygons_created++;

		if (below_count == 4) {

			p_res[polygons_created] = Face3(below[2], below[3], below[0]);
			p_is_point_over[polygons_created] = false;
			polygons_created++;
		}
	}

	return polygons_created;
}

bool Face3::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {

	return Geometry::ray_intersects_triangle(p_from, p_dir, vertex[0], vertex[1], vertex[2], p_intersection);
}

bool Face3::intersects_segment(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *p_intersection) const {

	return Geometry::segment_intersects_triangle(p_from, p_dir, vertex[0], vertex[1], vertex[2], p_intersection);
}

bool Face3::is_degenerate() const {

	Vector3 normal = vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]);
	return (normal.length_squared() < CMP_EPSILON2);
}

Face3::Side Face3::get_side_of(const Face3 &p_face, ClockDirection p_clock_dir) const {

	int over = 0, under = 0;

	Plane plane = get_plane(p_clock_dir);

	for (int i = 0; i < 3; i++) {

		const Vector3 &v = p_face.vertex[i];

		if (plane.has_point(v)) //coplanar, don't bother
			continue;

		if (plane.is_point_over(v))
			over++;
		else
			under++;
	}

	if (over > 0 && under == 0)
		return SIDE_OVER;
	else if (under > 0 && over == 0)
		return SIDE_UNDER;
	else if (under == 0 && over == 0)
		return SIDE_COPLANAR;
	else
		return SIDE_SPANNING;
}

Vector3 Face3::get_random_point_inside() const {

	real_t a = Math::random(0, 1);
	real_t b = Math::random(0, 1);
	if (a > b) {
		SWAP(a, b);
	}

	return vertex[0] * a + vertex[1] * (b - a) + vertex[2] * (1.0 - b);
}

Plane Face3::get_plane(ClockDirection p_dir) const {

	return Plane(vertex[0], vertex[1], vertex[2], p_dir);
}

Vector3 Face3::get_median_point() const {

	return (vertex[0] + vertex[1] + vertex[2]) / 3.0;
}

real_t Face3::get_area() const {

	return vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]).length();
}

ClockDirection Face3::get_clock_dir() const {

	Vector3 normal = vec3_cross(vertex[0] - vertex[1], vertex[0] - vertex[2]);
	//printf("normal is %g,%g,%g x %g,%g,%g- wtfu is %g\n",tofloat(normal.x),tofloat(normal.y),tofloat(normal.z),tofloat(vertex[0].x),tofloat(vertex[0].y),tofloat(vertex[0].z),tofloat( normal.dot( vertex[0] ) ) );
	return (normal.dot(vertex[0]) >= 0) ? CLOCKWISE : COUNTERCLOCKWISE;
}

bool Face3::intersects_aabb(const Rect3 &p_aabb) const {

	/** TEST PLANE **/
	if (!p_aabb.intersects_plane(get_plane()))
		return false;

/** TEST FACE AXIS */

#define TEST_AXIS(m_ax)                                       \
	{                                                         \
		real_t aabb_min = p_aabb.pos.m_ax;                    \
		real_t aabb_max = p_aabb.pos.m_ax + p_aabb.size.m_ax; \
		real_t tri_min, tri_max;                              \
		for (int i = 0; i < 3; i++) {                         \
			if (i == 0 || vertex[i].m_ax > tri_max)           \
				tri_max = vertex[i].m_ax;                     \
			if (i == 0 || vertex[i].m_ax < tri_min)           \
				tri_min = vertex[i].m_ax;                     \
		}                                                     \
                                                              \
		if (tri_max < aabb_min || aabb_max < tri_min)         \
			return false;                                     \
	}

	TEST_AXIS(x);
	TEST_AXIS(y);
	TEST_AXIS(z);

	/** TEST ALL EDGES **/

	Vector3 edge_norms[3] = {
		vertex[0] - vertex[1],
		vertex[1] - vertex[2],
		vertex[2] - vertex[0],
	};

	for (int i = 0; i < 12; i++) {

		Vector3 from, to;
		p_aabb.get_edge(i, from, to);
		Vector3 e1 = from - to;
		for (int j = 0; j < 3; j++) {
			Vector3 e2 = edge_norms[j];

			Vector3 axis = vec3_cross(e1, e2);

			if (axis.length_squared() < 0.0001)
				continue; // coplanar
			axis.normalize();

			real_t minA, maxA, minB, maxB;
			p_aabb.project_range_in_plane(Plane(axis, 0), minA, maxA);
			project_range(axis, Transform(), minB, maxB);

			if (maxA < minB || maxB < minA)
				return false;
		}
	}
	return true;
}

Face3::operator String() const {

	return String() + vertex[0] + ", " + vertex[1] + ", " + vertex[2];
}

void Face3::project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {

	for (int i = 0; i < 3; i++) {

		Vector3 v = p_transform.xform(vertex[i]);
		real_t d = p_normal.dot(v);

		if (i == 0 || d > r_max)
			r_max = d;

		if (i == 0 || d < r_min)
			r_min = d;
	}
}

void Face3::get_support(const Vector3 &p_normal, const Transform &p_transform, Vector3 *p_vertices, int *p_count, int p_max) const {

#define _FACE_IS_VALID_SUPPORT_TRESHOLD 0.98
#define _EDGE_IS_VALID_SUPPORT_TRESHOLD 0.05

	if (p_max <= 0)
		return;

	Vector3 n = p_transform.basis.xform_inv(p_normal);

	/** TEST FACE AS SUPPORT **/
	if (get_plane().normal.dot(n) > _FACE_IS_VALID_SUPPORT_TRESHOLD) {

		*p_count = MIN(3, p_max);

		for (int i = 0; i < *p_count; i++) {

			p_vertices[i] = p_transform.xform(vertex[i]);
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

		if (i != vert_support_idx && i + 1 != vert_support_idx)
			continue;

		// check if edge is valid as a support
		real_t dot = (vertex[i] - vertex[(i + 1) % 3]).normalized().dot(n);
		dot = ABS(dot);
		if (dot < _EDGE_IS_VALID_SUPPORT_TRESHOLD) {

			*p_count = MIN(2, p_max);

			for (int j = 0; j < *p_count; j++)
				p_vertices[j] = p_transform.xform(vertex[(j + i) % 3]);

			return;
		}
	}

	*p_count = 1;
	p_vertices[0] = p_transform.xform(vertex[vert_support_idx]);
}

Vector3 Face3::get_closest_point_to(const Vector3 &p_point) const {

	Vector3 edge0 = vertex[1] - vertex[0];
	Vector3 edge1 = vertex[2] - vertex[0];
	Vector3 v0 = vertex[0] - p_point;

	real_t a = edge0.dot(edge0);
	real_t b = edge0.dot(edge1);
	real_t c = edge1.dot(edge1);
	real_t d = edge0.dot(v0);
	real_t e = edge1.dot(v0);

	real_t det = a * c - b * b;
	real_t s = b * e - c * d;
	real_t t = b * d - a * e;

	if (s + t < det) {
		if (s < 0.f) {
			if (t < 0.f) {
				if (d < 0.f) {
					s = CLAMP(-d / a, 0.f, 1.f);
					t = 0.f;
				} else {
					s = 0.f;
					t = CLAMP(-e / c, 0.f, 1.f);
				}
			} else {
				s = 0.f;
				t = CLAMP(-e / c, 0.f, 1.f);
			}
		} else if (t < 0.f) {
			s = CLAMP(-d / a, 0.f, 1.f);
			t = 0.f;
		} else {
			real_t invDet = 1.f / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < 0.f) {
			real_t tmp0 = b + d;
			real_t tmp1 = c + e;
			if (tmp1 > tmp0) {
				real_t numer = tmp1 - tmp0;
				real_t denom = a - 2 * b + c;
				s = CLAMP(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				t = CLAMP(-e / c, 0.f, 1.f);
				s = 0.f;
			}
		} else if (t < 0.f) {
			if (a + d > b + e) {
				real_t numer = c + e - b - d;
				real_t denom = a - 2 * b + c;
				s = CLAMP(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				s = CLAMP(-e / c, 0.f, 1.f);
				t = 0.f;
			}
		} else {
			real_t numer = c + e - b - d;
			real_t denom = a - 2 * b + c;
			s = CLAMP(numer / denom, 0.f, 1.f);
			t = 1.f - s;
		}
	}

	return vertex[0] + s * edge0 + t * edge1;
}

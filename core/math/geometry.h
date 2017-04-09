/*************************************************************************/
/*  geometry.h                                                           */
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
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "dvector.h"
#include "face3.h"
#include "math_2d.h"
#include "object.h"
#include "print_string.h"
#include "triangulate.h"
#include "vector.h"
#include "vector3.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Geometry {
	Geometry();

public:
	static real_t get_closest_points_between_segments(const Vector2 &p1, const Vector2 &q1, const Vector2 &p2, const Vector2 &q2, Vector2 &c1, Vector2 &c2) {

		Vector2 d1 = q1 - p1; // Direction vector of segment S1
		Vector2 d2 = q2 - p2; // Direction vector of segment S2
		Vector2 r = p1 - p2;
		real_t a = d1.dot(d1); // Squared length of segment S1, always nonnegative
		real_t e = d2.dot(d2); // Squared length of segment S2, always nonnegative
		real_t f = d2.dot(r);
		real_t s, t;
		// Check if either or both segments degenerate into points
		if (a <= CMP_EPSILON && e <= CMP_EPSILON) {
			// Both segments degenerate into points
			c1 = p1;
			c2 = p2;
			return Math::sqrt((c1 - c2).dot(c1 - c2));
		}
		if (a <= CMP_EPSILON) {
			// First segment degenerates into a point
			s = 0.0;
			t = f / e; // s = 0 => t = (b*s + f) / e = f / e
			t = CLAMP(t, 0.0, 1.0);
		} else {
			real_t c = d1.dot(r);
			if (e <= CMP_EPSILON) {
				// Second segment degenerates into a point
				t = 0.0;
				s = CLAMP(-c / a, 0.0, 1.0); // t = 0 => s = (b*t - c) / a = -c / a
			} else {
				// The general nondegenerate case starts here
				real_t b = d1.dot(d2);
				real_t denom = a * e - b * b; // Always nonnegative
				// If segments not parallel, compute closest point on L1 to L2 and
				// clamp to segment S1. Else pick arbitrary s (here 0)
				if (denom != 0.0) {
					s = CLAMP((b * f - c * e) / denom, 0.0, 1.0);
				} else
					s = 0.0;
				// Compute point on L2 closest to S1(s) using
				// t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e
				t = (b * s + f) / e;

				//If t in [0,1] done. Else clamp t, recompute s for the new value
				// of t using s = Dot((P2 + D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a
				// and clamp s to [0, 1]
				if (t < 0.0) {
					t = 0.0;
					s = CLAMP(-c / a, 0.0, 1.0);
				} else if (t > 1.0) {
					t = 1.0;
					s = CLAMP((b - c) / a, 0.0, 1.0);
				}
			}
		}
		c1 = p1 + d1 * s;
		c2 = p2 + d2 * t;
		return Math::sqrt((c1 - c2).dot(c1 - c2));
	}

	static void get_closest_points_between_segments(const Vector3 &p1, const Vector3 &p2, const Vector3 &q1, const Vector3 &q2, Vector3 &c1, Vector3 &c2) {
#if 1
//do the function 'd' as defined by pb. I think is is dot product of some sort
#define d_of(m, n, o, p) ((m.x - n.x) * (o.x - p.x) + (m.y - n.y) * (o.y - p.y) + (m.z - n.z) * (o.z - p.z))

		//calculate the parametric position on the 2 curves, mua and mub
		real_t mua = (d_of(p1, q1, q2, q1) * d_of(q2, q1, p2, p1) - d_of(p1, q1, p2, p1) * d_of(q2, q1, q2, q1)) / (d_of(p2, p1, p2, p1) * d_of(q2, q1, q2, q1) - d_of(q2, q1, p2, p1) * d_of(q2, q1, p2, p1));
		real_t mub = (d_of(p1, q1, q2, q1) + mua * d_of(q2, q1, p2, p1)) / d_of(q2, q1, q2, q1);

		//clip the value between [0..1] constraining the solution to lie on the original curves
		if (mua < 0) mua = 0;
		if (mub < 0) mub = 0;
		if (mua > 1) mua = 1;
		if (mub > 1) mub = 1;
		c1 = p1.linear_interpolate(p2, mua);
		c2 = q1.linear_interpolate(q2, mub);
#else
		//this is broken do not use
		Vector3 u = p2 - p1;
		Vector3 v = q2 - q1;
		Vector3 w = p1 - q1;
		float a = u.dot(u);
		float b = u.dot(v);
		float c = v.dot(v); // always >= 0
		float d = u.dot(w);
		float e = v.dot(w);
		float D = a * c - b * b; // always >= 0
		float sc, tc;

		// compute the line parameters of the two closest points
		if (D < CMP_EPSILON) { // the lines are almost parallel
			sc = 0.0;
			tc = (b > c ? d / b : e / c); // use the largest denominator
		} else {
			sc = (b * e - c * d) / D;
			tc = (a * e - b * d) / D;
		}

		c1 = w + sc * u;
		c2 = w + tc * v;
// get the difference of the two closest points
//Vector   dP = w + (sc * u) - (tc * v);  // =  L1(sc) - L2(tc)
#endif
	}

	static real_t get_closest_distance_between_segments(const Vector3 &p_from_a, const Vector3 &p_to_a, const Vector3 &p_from_b, const Vector3 &p_to_b) {
		Vector3 u = p_to_a - p_from_a;
		Vector3 v = p_to_b - p_from_b;
		Vector3 w = p_from_a - p_to_a;
		real_t a = u.dot(u); // always >= 0
		real_t b = u.dot(v);
		real_t c = v.dot(v); // always >= 0
		real_t d = u.dot(w);
		real_t e = v.dot(w);
		real_t D = a * c - b * b; // always >= 0
		real_t sc, sN, sD = D; // sc = sN / sD, default sD = D >= 0
		real_t tc, tN, tD = D; // tc = tN / tD, default tD = D >= 0

		// compute the line parameters of the two closest points
		if (D < CMP_EPSILON) { // the lines are almost parallel
			sN = 0.0; // force using point P0 on segment S1
			sD = 1.0; // to prevent possible division by 0.0 later
			tN = e;
			tD = c;
		} else { // get the closest points on the infinite lines
			sN = (b * e - c * d);
			tN = (a * e - b * d);
			if (sN < 0.0) { // sc < 0 => the s=0 edge is visible
				sN = 0.0;
				tN = e;
				tD = c;
			} else if (sN > sD) { // sc > 1 => the s=1 edge is visible
				sN = sD;
				tN = e + b;
				tD = c;
			}
		}

		if (tN < 0.0) { // tc < 0 => the t=0 edge is visible
			tN = 0.0;
			// recompute sc for this edge
			if (-d < 0.0)
				sN = 0.0;
			else if (-d > a)
				sN = sD;
			else {
				sN = -d;
				sD = a;
			}
		} else if (tN > tD) { // tc > 1 => the t=1 edge is visible
			tN = tD;
			// recompute sc for this edge
			if ((-d + b) < 0.0)
				sN = 0;
			else if ((-d + b) > a)
				sN = sD;
			else {
				sN = (-d + b);
				sD = a;
			}
		}
		// finally do the division to get sc and tc
		sc = (Math::abs(sN) < CMP_EPSILON ? 0.0 : sN / sD);
		tc = (Math::abs(tN) < CMP_EPSILON ? 0.0 : tN / tD);

		// get the difference of the two closest points
		Vector3 dP = w + (sc * u) - (tc * v); // = S1(sc) - S2(tc)

		return dP.length(); // return the closest distance
	}

	static inline bool ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = 0) {
		Vector3 e1 = p_v1 - p_v0;
		Vector3 e2 = p_v2 - p_v0;
		Vector3 h = p_dir.cross(e2);
		real_t a = e1.dot(h);
		if (a > -CMP_EPSILON && a < CMP_EPSILON) // parallel test
			return false;

		real_t f = 1.0 / a;

		Vector3 s = p_from - p_v0;
		real_t u = f * s.dot(h);

		if (u < 0.0 || u > 1.0)
			return false;

		Vector3 q = s.cross(e1);

		real_t v = f * p_dir.dot(q);

		if (v < 0.0 || u + v > 1.0)
			return false;

		// at this stage we can compute t to find out where
		// the intersection point is on the line
		real_t t = f * e2.dot(q);

		if (t > 0.00001) { // ray intersection
			if (r_res)
				*r_res = p_from + p_dir * t;
			return true;
		} else // this means that there is a line intersection
			// but not a ray intersection
			return false;
	}

	static inline bool segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = 0) {

		Vector3 rel = p_to - p_from;
		Vector3 e1 = p_v1 - p_v0;
		Vector3 e2 = p_v2 - p_v0;
		Vector3 h = rel.cross(e2);
		real_t a = e1.dot(h);
		if (a > -CMP_EPSILON && a < CMP_EPSILON) // parallel test
			return false;

		real_t f = 1.0 / a;

		Vector3 s = p_from - p_v0;
		real_t u = f * s.dot(h);

		if (u < 0.0 || u > 1.0)
			return false;

		Vector3 q = s.cross(e1);

		real_t v = f * rel.dot(q);

		if (v < 0.0 || u + v > 1.0)
			return false;

		// at this stage we can compute t to find out where
		// the intersection point is on the line
		real_t t = f * e2.dot(q);

		if (t > CMP_EPSILON && t <= 1.0) { // ray intersection
			if (r_res)
				*r_res = p_from + rel * t;
			return true;
		} else // this means that there is a line intersection
			// but not a ray intersection
			return false;
	}

	static inline bool segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 *r_res = 0, Vector3 *r_norm = 0) {

		Vector3 sphere_pos = p_sphere_pos - p_from;
		Vector3 rel = (p_to - p_from);
		real_t rel_l = rel.length();
		if (rel_l < CMP_EPSILON)
			return false; // both points are the same
		Vector3 normal = rel / rel_l;

		real_t sphere_d = normal.dot(sphere_pos);

		//Vector3 ray_closest=normal*sphere_d;

		real_t ray_distance = sphere_pos.distance_to(normal * sphere_d);

		if (ray_distance >= p_sphere_radius)
			return false;

		real_t inters_d2 = p_sphere_radius * p_sphere_radius - ray_distance * ray_distance;
		real_t inters_d = sphere_d;

		if (inters_d2 >= CMP_EPSILON)
			inters_d -= Math::sqrt(inters_d2);

		// check in segment
		if (inters_d < 0 || inters_d > rel_l)
			return false;

		Vector3 result = p_from + normal * inters_d;

		if (r_res)
			*r_res = result;
		if (r_norm)
			*r_norm = (result - p_sphere_pos).normalized();

		return true;
	}

	static inline bool segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, real_t p_height, real_t p_radius, Vector3 *r_res = 0, Vector3 *r_norm = 0) {

		Vector3 rel = (p_to - p_from);
		real_t rel_l = rel.length();
		if (rel_l < CMP_EPSILON)
			return false; // both points are the same

		// first check if they are parallel
		Vector3 normal = (rel / rel_l);
		Vector3 crs = normal.cross(Vector3(0, 0, 1));
		real_t crs_l = crs.length();

		Vector3 z_dir;

		if (crs_l < CMP_EPSILON) {
			//blahblah parallel
			z_dir = Vector3(1, 0, 0); //any x/y vector ok
		} else {
			z_dir = crs / crs_l;
		}

		real_t dist = z_dir.dot(p_from);

		if (dist >= p_radius)
			return false; // too far away

		// convert to 2D
		real_t w2 = p_radius * p_radius - dist * dist;
		if (w2 < CMP_EPSILON)
			return false; //avoid numerical error
		Size2 size(Math::sqrt(w2), p_height * 0.5);

		Vector3 x_dir = z_dir.cross(Vector3(0, 0, 1)).normalized();

		Vector2 from2D(x_dir.dot(p_from), p_from.z);
		Vector2 to2D(x_dir.dot(p_to), p_to.z);

		real_t min = 0, max = 1;

		int axis = -1;

		for (int i = 0; i < 2; i++) {

			real_t seg_from = from2D[i];
			real_t seg_to = to2D[i];
			real_t box_begin = -size[i];
			real_t box_end = size[i];
			real_t cmin, cmax;

			if (seg_from < seg_to) {

				if (seg_from > box_end || seg_to < box_begin)
					return false;
				real_t length = seg_to - seg_from;
				cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
				cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;

			} else {

				if (seg_to > box_end || seg_from < box_begin)
					return false;
				real_t length = seg_to - seg_from;
				cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
				cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
			}

			if (cmin > min) {
				min = cmin;
				axis = i;
			}
			if (cmax < max)
				max = cmax;
			if (max < min)
				return false;
		}

		// convert to 3D again
		Vector3 result = p_from + (rel * min);
		Vector3 res_normal = result;

		if (axis == 0) {
			res_normal.z = 0;
		} else {
			res_normal.x = 0;
			res_normal.y = 0;
		}

		res_normal.normalize();

		if (r_res)
			*r_res = result;
		if (r_norm)
			*r_norm = res_normal;

		return true;
	}

	static bool segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Plane *p_planes, int p_plane_count, Vector3 *p_res, Vector3 *p_norm) {

		real_t min = -1e20, max = 1e20;

		Vector3 rel = p_to - p_from;
		real_t rel_l = rel.length();

		if (rel_l < CMP_EPSILON)
			return false;

		Vector3 dir = rel / rel_l;

		int min_index = -1;

		for (int i = 0; i < p_plane_count; i++) {

			const Plane &p = p_planes[i];

			real_t den = p.normal.dot(dir);

			//printf("den is %i\n",den);
			if (Math::abs(den) <= CMP_EPSILON)
				continue; // ignore parallel plane

			real_t dist = -p.distance_to(p_from) / den;

			if (den > 0) {
				//backwards facing plane
				if (dist < max)
					max = dist;
			} else {

				//front facing plane
				if (dist > min) {
					min = dist;
					min_index = i;
				}
			}
		}

		if (max <= min || min < 0 || min > rel_l || min_index == -1) // exit conditions
			return false; // no intersection

		if (p_res)
			*p_res = p_from + dir * min;
		if (p_norm)
			*p_norm = p_planes[min_index].normal;

		return true;
	}

	static Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 *p_segment) {

		Vector3 p = p_point - p_segment[0];
		Vector3 n = p_segment[1] - p_segment[0];
		real_t l = n.length();
		if (l < 1e-10)
			return p_segment[0]; // both points are the same, just give any
		n /= l;

		real_t d = n.dot(p);

		if (d <= 0.0)
			return p_segment[0]; // before first point
		else if (d >= l)
			return p_segment[1]; // after first point
		else
			return p_segment[0] + n * d; // inside
	}

	static Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 *p_segment) {

		Vector3 p = p_point - p_segment[0];
		Vector3 n = p_segment[1] - p_segment[0];
		real_t l = n.length();
		if (l < 1e-10)
			return p_segment[0]; // both points are the same, just give any
		n /= l;

		real_t d = n.dot(p);

		return p_segment[0] + n * d; // inside
	}

	static Vector2 get_closest_point_to_segment_2d(const Vector2 &p_point, const Vector2 *p_segment) {

		Vector2 p = p_point - p_segment[0];
		Vector2 n = p_segment[1] - p_segment[0];
		real_t l = n.length();
		if (l < 1e-10)
			return p_segment[0]; // both points are the same, just give any
		n /= l;

		real_t d = n.dot(p);

		if (d <= 0.0)
			return p_segment[0]; // before first point
		else if (d >= l)
			return p_segment[1]; // after first point
		else
			return p_segment[0] + n * d; // inside
	}

	static bool is_point_in_triangle(const Vector2 &s, const Vector2 &a, const Vector2 &b, const Vector2 &c) {
		int as_x = s.x - a.x;
		int as_y = s.y - a.y;

		bool s_ab = (b.x - a.x) * as_y - (b.y - a.y) * as_x > 0;

		if (((c.x - a.x) * as_y - (c.y - a.y) * as_x > 0) == s_ab) return false;

		if (((c.x - b.x) * (s.y - b.y) - (c.y - b.y) * (s.x - b.x) > 0) != s_ab) return false;

		return true;
	}
	static Vector2 get_closest_point_to_segment_uncapped_2d(const Vector2 &p_point, const Vector2 *p_segment) {

		Vector2 p = p_point - p_segment[0];
		Vector2 n = p_segment[1] - p_segment[0];
		real_t l = n.length();
		if (l < 1e-10)
			return p_segment[0]; // both points are the same, just give any
		n /= l;

		real_t d = n.dot(p);

		return p_segment[0] + n * d; // inside
	}

	static bool segment_intersects_segment_2d(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b, Vector2 *r_result) {

		Vector2 B = p_to_a - p_from_a;
		Vector2 C = p_from_b - p_from_a;
		Vector2 D = p_to_b - p_from_a;

		real_t ABlen = B.dot(B);
		if (ABlen <= 0)
			return false;
		Vector2 Bn = B / ABlen;
		C = Vector2(C.x * Bn.x + C.y * Bn.y, C.y * Bn.x - C.x * Bn.y);
		D = Vector2(D.x * Bn.x + D.y * Bn.y, D.y * Bn.x - D.x * Bn.y);

		if ((C.y < 0 && D.y < 0) || (C.y >= 0 && D.y >= 0))
			return false;

		real_t ABpos = D.x + (C.x - D.x) * D.y / (D.y - C.y);

		//  Fail if segment C-D crosses line A-B outside of segment A-B.
		if (ABpos < 0 || ABpos > 1.0)
			return false;

		//  (4) Apply the discovered position to line A-B in the original coordinate system.
		if (r_result)
			*r_result = p_from_a + B * ABpos;

		return true;
	}

	static inline bool point_in_projected_triangle(const Vector3 &p_point, const Vector3 &p_v1, const Vector3 &p_v2, const Vector3 &p_v3) {

		Vector3 face_n = (p_v1 - p_v3).cross(p_v1 - p_v2);

		Vector3 n1 = (p_point - p_v3).cross(p_point - p_v2);

		if (face_n.dot(n1) < 0)
			return false;

		Vector3 n2 = (p_v1 - p_v3).cross(p_v1 - p_point);

		if (face_n.dot(n2) < 0)
			return false;

		Vector3 n3 = (p_v1 - p_point).cross(p_v1 - p_v2);

		if (face_n.dot(n3) < 0)
			return false;

		return true;
	}

	static inline bool triangle_sphere_intersection_test(const Vector3 *p_triangle, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact) {

		real_t d = p_normal.dot(p_sphere_pos) - p_normal.dot(p_triangle[0]);

		if (d > p_sphere_radius || d < -p_sphere_radius) // not touching the plane of the face, return
			return false;

		Vector3 contact = p_sphere_pos - (p_normal * d);

		/** 2nd) TEST INSIDE TRIANGLE **/

		if (Geometry::point_in_projected_triangle(contact, p_triangle[0], p_triangle[1], p_triangle[2])) {
			r_triangle_contact = contact;
			r_sphere_contact = p_sphere_pos - p_normal * p_sphere_radius;
			//printf("solved inside triangle\n");
			return true;
		}

		/** 3rd TEST INSIDE EDGE CYLINDERS **/

		const Vector3 verts[4] = { p_triangle[0], p_triangle[1], p_triangle[2], p_triangle[0] }; // for() friendly

		for (int i = 0; i < 3; i++) {

			// check edge cylinder

			Vector3 n1 = verts[i] - verts[i + 1];
			Vector3 n2 = p_sphere_pos - verts[i + 1];

			///@TODO i could discard by range here to make the algorithm quicker? dunno..

			// check point within cylinder radius
			Vector3 axis = n1.cross(n2).cross(n1);
			axis.normalize(); // ugh

			real_t ad = axis.dot(n2);

			if (ABS(ad) > p_sphere_radius) {
				// no chance with this edge, too far away
				continue;
			}

			// check point within edge capsule cylinder
			/** 4th TEST INSIDE EDGE POINTS **/

			real_t sphere_at = n1.dot(n2);

			if (sphere_at >= 0 && sphere_at < n1.dot(n1)) {

				r_triangle_contact = p_sphere_pos - axis * (axis.dot(n2));
				r_sphere_contact = p_sphere_pos - axis * p_sphere_radius;
				// point inside here
				//printf("solved inside edge\n");
				return true;
			}

			real_t r2 = p_sphere_radius * p_sphere_radius;

			if (n2.length_squared() < r2) {

				Vector3 n = (p_sphere_pos - verts[i + 1]).normalized();

				//r_triangle_contact=verts[i+1]+n*p_sphere_radius;p_sphere_pos+axis*(p_sphere_radius-axis.dot(n2));
				r_triangle_contact = verts[i + 1];
				r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
				//printf("solved inside point segment 1\n");
				return true;
			}

			if (n2.distance_squared_to(n1) < r2) {
				Vector3 n = (p_sphere_pos - verts[i]).normalized();

				//r_triangle_contact=verts[i]+n*p_sphere_radius;p_sphere_pos+axis*(p_sphere_radius-axis.dot(n2));
				r_triangle_contact = verts[i];
				r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
				//printf("solved inside point segment 1\n");
				return true;
			}

			break; // It's pointless to continue at this point, so save some cpu cycles
		}

		return false;
	}

	static real_t segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius) {

		Vector2 line_vec = p_to - p_from;
		Vector2 vec_to_line = p_from - p_circle_pos;

		/* create a quadratic formula of the form ax^2 + bx + c = 0 */
		real_t a, b, c;

		a = line_vec.dot(line_vec);
		b = 2 * vec_to_line.dot(line_vec);
		c = vec_to_line.dot(vec_to_line) - p_circle_radius * p_circle_radius;

		/* solve for t */
		real_t sqrtterm = b * b - 4 * a * c;

		/* if the term we intend to square root is less than 0 then the answer won't be real, so it definitely won't be t in the range 0 to 1 */
		if (sqrtterm < 0) return -1;

		/* if we can assume that the line segment starts outside the circle (e.g. for continuous time collision detection) then the following can be skipped and we can just return the equivalent of res1 */
		sqrtterm = Math::sqrt(sqrtterm);
		real_t res1 = (-b - sqrtterm) / (2 * a);
		//real_t res2 = ( -b + sqrtterm ) / (2 * a);

		return (res1 >= 0 && res1 <= 1) ? res1 : -1;
	}

	static inline Vector<Vector3> clip_polygon(const Vector<Vector3> &polygon, const Plane &p_plane) {

		enum LocationCache {
			LOC_INSIDE = 1,
			LOC_BOUNDARY = 0,
			LOC_OUTSIDE = -1
		};

		if (polygon.size() == 0)
			return polygon;

		int *location_cache = (int *)alloca(sizeof(int) * polygon.size());
		int inside_count = 0;
		int outside_count = 0;

		for (int a = 0; a < polygon.size(); a++) {
			//real_t p_plane.d = (*this) * polygon[a];
			real_t dist = p_plane.distance_to(polygon[a]);
			if (dist < -CMP_POINT_IN_PLANE_EPSILON) {
				location_cache[a] = LOC_INSIDE;
				inside_count++;
			} else {
				if (dist > CMP_POINT_IN_PLANE_EPSILON) {
					location_cache[a] = LOC_OUTSIDE;
					outside_count++;
				} else {
					location_cache[a] = LOC_BOUNDARY;
				}
			}
		}

		if (outside_count == 0) {

			return polygon; // no changes

		} else if (inside_count == 0) {

			return Vector<Vector3>(); //empty
		}

		long previous = polygon.size() - 1;
		Vector<Vector3> clipped;

		for (int index = 0; index < polygon.size(); index++) {
			int loc = location_cache[index];
			if (loc == LOC_OUTSIDE) {
				if (location_cache[previous] == LOC_INSIDE) {
					const Vector3 &v1 = polygon[previous];
					const Vector3 &v2 = polygon[index];

					Vector3 segment = v1 - v2;
					real_t den = p_plane.normal.dot(segment);
					real_t dist = p_plane.distance_to(v1) / den;
					dist = -dist;
					clipped.push_back(v1 + segment * dist);
				}
			} else {
				const Vector3 &v1 = polygon[index];
				if ((loc == LOC_INSIDE) && (location_cache[previous] == LOC_OUTSIDE)) {
					const Vector3 &v2 = polygon[previous];
					Vector3 segment = v1 - v2;
					real_t den = p_plane.normal.dot(segment);
					real_t dist = p_plane.distance_to(v1) / den;
					dist = -dist;
					clipped.push_back(v1 + segment * dist);
				}

				clipped.push_back(v1);
			}

			previous = index;
		}

		return clipped;
	}

	static Vector<int> triangulate_polygon(const Vector<Vector2> &p_polygon) {

		Vector<int> triangles;
		if (!Triangulate::triangulate(p_polygon, triangles))
			return Vector<int>(); //fail
		return triangles;
	}

	static Vector<Vector<Vector2> > (*_decompose_func)(const Vector<Vector2> &p_polygon);
	static Vector<Vector<Vector2> > decompose_polygon(const Vector<Vector2> &p_polygon) {

		if (_decompose_func)
			return _decompose_func(p_polygon);

		return Vector<Vector<Vector2> >();
	}

	static PoolVector<PoolVector<Face3> > separate_objects(PoolVector<Face3> p_array);

	static PoolVector<Face3> wrap_geometry(PoolVector<Face3> p_array, real_t *p_error = NULL); ///< create a "wrap" that encloses the given geometry

	struct MeshData {

		struct Face {
			Plane plane;
			Vector<int> indices;
		};

		Vector<Face> faces;

		struct Edge {

			int a, b;
		};

		Vector<Edge> edges;

		Vector<Vector3> vertices;

		void optimize_vertices();
	};

	_FORCE_INLINE_ static int get_uv84_normal_bit(const Vector3 &p_vector) {

		int lat = Math::fast_ftoi(Math::floor(Math::acos(p_vector.dot(Vector3(0, 1, 0))) * 4.0 / Math_PI + 0.5));

		if (lat == 0) {
			return 24;
		} else if (lat == 4) {
			return 25;
		}

		int lon = Math::fast_ftoi(Math::floor((Math_PI + Math::atan2(p_vector.x, p_vector.z)) * 8.0 / (Math_PI * 2.0) + 0.5)) % 8;

		return lon + (lat - 1) * 8;
	}

	_FORCE_INLINE_ static int get_uv84_normal_bit_neighbors(int p_idx) {

		if (p_idx == 24) {
			return 1 | 2 | 4 | 8;
		} else if (p_idx == 25) {
			return (1 << 23) | (1 << 22) | (1 << 21) | (1 << 20);
		} else {

			int ret = 0;
			if ((p_idx % 8) == 0)
				ret |= (1 << (p_idx + 7));
			else
				ret |= (1 << (p_idx - 1));
			if ((p_idx % 8) == 7)
				ret |= (1 << (p_idx - 7));
			else
				ret |= (1 << (p_idx + 1));

			int mask = ret | (1 << p_idx);
			if (p_idx < 8)
				ret |= 24;
			else
				ret |= mask >> 8;

			if (p_idx >= 16)
				ret |= 25;
			else
				ret |= mask << 8;

			return ret;
		}
	}

	static real_t vec2_cross(const Point2 &O, const Point2 &A, const Point2 &B) {
		return (real_t)(A.x - O.x) * (B.y - O.y) - (real_t)(A.y - O.y) * (B.x - O.x);
	}

	// Returns a list of points on the convex hull in counter-clockwise order.
	// Note: the last point in the returned list is the same as the first one.
	static Vector<Point2> convex_hull_2d(Vector<Point2> P) {
		int n = P.size(), k = 0;
		Vector<Point2> H;
		H.resize(2 * n);

		// Sort points lexicographically
		P.sort();

		// Build lower hull
		for (int i = 0; i < n; ++i) {
			while (k >= 2 && vec2_cross(H[k - 2], H[k - 1], P[i]) <= 0)
				k--;
			H[k++] = P[i];
		}

		// Build upper hull
		for (int i = n - 2, t = k + 1; i >= 0; i--) {
			while (k >= t && vec2_cross(H[k - 2], H[k - 1], P[i]) <= 0)
				k--;
			H[k++] = P[i];
		}

		H.resize(k);
		return H;
	}

	static MeshData build_convex_mesh(const PoolVector<Plane> &p_planes);
	static PoolVector<Plane> build_sphere_planes(real_t p_radius, int p_lats, int p_lons, Vector3::Axis p_axis = Vector3::AXIS_Z);
	static PoolVector<Plane> build_box_planes(const Vector3 &p_extents);
	static PoolVector<Plane> build_cylinder_planes(real_t p_radius, real_t p_height, int p_sides, Vector3::Axis p_axis = Vector3::AXIS_Z);
	static PoolVector<Plane> build_capsule_planes(real_t p_radius, real_t p_height, int p_sides, int p_lats, Vector3::Axis p_axis = Vector3::AXIS_Z);

	static void make_atlas(const Vector<Size2i> &p_rects, Vector<Point2i> &r_result, Size2i &r_size);
};

#endif

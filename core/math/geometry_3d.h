/*************************************************************************/
/*  geometry_3d.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GEOMETRY_3D_H
#define GEOMETRY_3D_H

#include "core/math/face3.h"
#include "core/object/object.h"
#include "core/templates/vector.h"

class Geometry3D {
public:
// Dot product of NM and PO vectors
#define dot_4(M, N, O, P) ((M.x - N.x) * (O.x - P.x) + (M.y - N.y) * (O.y - P.y) + (M.z - N.z) * (O.z - P.z))

// Returns t where A + t * AB is the closest point on the segment to point P (t can be outside [0..1], it's uncapped).
// it basically does this: (AB•AP) / (|AB|^2)
#define CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(P, A, B) (dot_4(A, B, A, P) / dot_4(A, B, A, B))

	static Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 *p_segment) {
		real_t t = 0.0f;

		if (p_segment[0].distance_squared_to(p_segment[1]) > 1e-20f) {
			t = CLAMP(CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_point, p_segment[0], p_segment[1]), 0, 1);
		}

		return p_segment[0].lerp(p_segment[1], t);
	}

	static Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 *p_segment) {
		real_t t = 0.0f;

		if (p_segment[0].distance_squared_to(p_segment[1]) > 1e-20f) {
			t = CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_point, p_segment[0], p_segment[1]);
		}

		return p_segment[0].lerp(p_segment[1], t);
	}

	static void get_closest_points_between_segments(const Vector3 &p_A, const Vector3 &p_B, const Vector3 &p_C, const Vector3 &p_D, Vector3 &r_c1, Vector3 &r_c2) {
		Vector3 a = p_B - p_A;
		Vector3 b = p_D - p_C;
		Vector3 c = p_C - p_A;

		real_t ta = 0.0f, tb = 0.0f;

		// If segments are not co-planar
		bool are_coplanar = Math::is_zero_approx((p_C - p_A).cross(p_D - p_A).dot(p_B - p_A));
		if (!are_coplanar) {
			// Segments AB and CD are not parallel
			// We know that the shortest segment (if not bounded) between segments (or rather, lines)
			// is going to be orthogonal to both of the lines. That means that it will be parallel to
			// the cross product of AB and CD.
			Vector3 n = a.cross(b);
			// We don't care about the length of n, we just want the direction, but since normalization
			// is an expensive calculation, involving square roots, we'll just get rid of the length later.

			// Now we can project any vector connecting the two segments (since it doesn't matter, we'll
			// use vector CA) on vector n to get the closest possible distance between segments. Since n
			// is not normalized, this is what we need:  (n / |n|) • CA = (n • CA) / |n|
			// But we don't want just the length of the shortest possible segment connecting lines AB and
			// CD, we want the segment itself. We do have both, the length and the direction, so we just
			// multiply the unit vector with the direction (n / |n|) with the length ((n • CA) / |n|) to
			// get:  ((n • CA) / |n|) * (n / |n|) = n * (n • CA) / |n|^2
			Vector3 s = n * (n.dot(p_A - p_C) / n.length_squared()); // s stands for  shortest possible

			// Now, we know that there exist such t_a and t_b that satisfy the equation:
			//  A + AB * t_a + s = C + CD * t_b
			// we can (kinda) simplify this equation by substituting variables a, b, c in:
			//  a * t_a + s = b * t_b + c  <=>  a * t_a - b * t_b = c - s
			// if you expand write the above in terms of the coordinates, you get what's written bellow.

			Vector3 axb = a.cross(b);
			Vector3 gxb = (c - s).cross(b);
			Vector3 gxa = (c - s).cross(a);

			if (!Math::is_zero_approx(axb.x)) {
				ta = gxb.x / axb.x;
				tb = gxa.x / axb.x;
			} else if (!Math::is_zero_approx(axb.y)) {
				ta = gxb.y / axb.y;
				tb = gxa.y / axb.y;
			} else if (!Math::is_zero_approx(axb.z)) {
				ta = gxb.z / axb.z;
				tb = gxa.z / axb.z;
			}

			ta = CLAMP(ta, 0, 1);
			tb = CLAMP(tb, 0, 1);
		}

		// Check all points manually
		real_t min_d = (p_A.lerp(p_B, ta)).distance_squared_to(p_C.lerp(p_D, tb));
		real_t d = 0;
		real_t t = 0;

		t = CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_A, p_C, p_D);
		t = CLAMP(t, 0, 1);
		d = p_A.distance_squared_to(p_C.lerp(p_D, t));
		if (d < min_d) {
			min_d = d;
			ta = 0;
			tb = t;
		}

		t = CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_B, p_C, p_D);
		t = CLAMP(t, 0, 1);
		d = p_B.distance_squared_to(p_C.lerp(p_D, t));
		if (d < min_d) {
			min_d = d;
			ta = 1;
			tb = t;
		}

		t = CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_C, p_A, p_B);
		t = CLAMP(t, 0, 1);
		d = p_C.distance_squared_to(p_A.lerp(p_B, t));
		if (d < min_d) {
			min_d = d;
			ta = t;
			tb = 0;
		}

		t = CLOSEST_POINT_ON_SEGMENT_PARAMETRIC(p_D, p_A, p_B);
		t = CLAMP(t, 0, 1);
		d = p_D.distance_squared_to(p_A.lerp(p_B, t));
		if (d < min_d) {
			min_d = d;
			ta = t;
			tb = 1;
		}

		r_c1 = p_A + a * ta;
		r_c2 = p_C + b * tb;
	}

#undef CLOSEST_POINT_ON_SEGMENT_PARAMETRIC
#undef dot_4

	static real_t get_closest_distance_between_segments(const Vector3 &p_a1, const Vector3 &p_a2, const Vector3 &p_b1, const Vector3 &p_b2) {
		Vector3 u = p_a1;
		Vector3 v = p_b1;

		get_closest_points_between_segments(p_a1, p_a2, p_b1, p_b2, u, v);

		return u.distance_to(v);
	}

	static inline bool ray_intersects_triangle_parametric(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, real_t *r_res = nullptr) {
		Vector3 e1 = p_v1 - p_v0;
		Vector3 e2 = p_v2 - p_v0;
		Vector3 h = p_dir.cross(e2);
		real_t a = e1.dot(h);
		if (Math::is_zero_approx(a)) { // Parallel test.
			return false;
		}

		real_t f = 1.0f / a;

		Vector3 s = p_from - p_v0;
		real_t u = f * s.dot(h);

		if ((u < 0.0f) || (u > 1.0f)) {
			return false;
		}

		Vector3 q = s.cross(e1);

		real_t v = f * p_dir.dot(q);

		if ((v < 0.0f) || (u + v > 1.0f)) {
			return false;
		}

		// At this stage we can compute t to find out where
		// the intersection point is on the line.
		real_t t = f * e2.dot(q);

		if (t > (real_t)CMP_EPSILON) { // ray intersection
			if (r_res) {
				*r_res = t;
			}
			return true;
		} else { // This means that there is a line intersection but not a ray intersection.
			return false;
		}
	}

	static inline bool ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = nullptr) {
		real_t t = 0.0f;
		bool intersects = ray_intersects_triangle_parametric(p_from, p_dir, p_v0, p_v1, p_v2, &t);

		if (intersects && r_res) {
			*r_res = p_from + p_dir * t;
		}

		return intersects;
	}

	static inline bool segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = nullptr) {
		Vector3 rel = p_to - p_from;

		real_t t = 0.0f;
		bool intersects = ray_intersects_triangle_parametric(p_from, rel, p_v0, p_v1, p_v2, &t);

		if (t <= 1.0f) {
			if (r_res) {
				*r_res = p_from + rel * t;
			}
			return intersects;
		}

		return false;
	}

	static inline bool segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 *r_res = nullptr, Vector3 *r_norm = nullptr) {
		Vector3 sphere_pos = p_sphere_pos - p_from;
		Vector3 rel = (p_to - p_from);
		real_t rel_l = rel.length();
		if (rel_l < (real_t)CMP_EPSILON) {
			return false; // Both points are the same.
		}
		Vector3 normal = rel / rel_l;

		real_t sphere_d = normal.dot(sphere_pos);

		real_t ray_distance = sphere_pos.distance_to(normal * sphere_d);

		if (ray_distance >= p_sphere_radius) {
			return false;
		}

		real_t inters_d2 = p_sphere_radius * p_sphere_radius - ray_distance * ray_distance;
		real_t inters_d = sphere_d;

		if (inters_d2 >= (real_t)CMP_EPSILON) {
			inters_d -= Math::sqrt(inters_d2);
		}

		// Check in segment.
		if (inters_d < 0 || inters_d > rel_l) {
			return false;
		}

		Vector3 result = p_from + normal * inters_d;

		if (r_res) {
			*r_res = result;
		}
		if (r_norm) {
			*r_norm = (result - p_sphere_pos).normalized();
		}

		return true;
	}

	static inline bool segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, real_t p_height, real_t p_radius, Vector3 *r_res = nullptr, Vector3 *r_norm = nullptr, int p_cylinder_axis = 2) {
		Vector3 rel = (p_to - p_from);
		real_t rel_l = rel.length();
		if (rel_l < (real_t)CMP_EPSILON) {
			return false; // Both points are the same.
		}

		ERR_FAIL_COND_V(p_cylinder_axis < 0, false);
		ERR_FAIL_COND_V(p_cylinder_axis > 2, false);
		Vector3 cylinder_axis;
		cylinder_axis[p_cylinder_axis] = 1.0f;

		// First check if they are parallel.
		Vector3 normal = (rel / rel_l);
		Vector3 crs = normal.cross(cylinder_axis);
		real_t crs_l = crs.length();

		Vector3 axis_dir;

		if (crs_l < (real_t)CMP_EPSILON) {
			Vector3 side_axis;
			side_axis[(p_cylinder_axis + 1) % 3] = 1.0f; // Any side axis OK.
			axis_dir = side_axis;
		} else {
			axis_dir = crs / crs_l;
		}

		real_t dist = axis_dir.dot(p_from);

		if (dist >= p_radius) {
			return false; // Too far away.
		}

		// Convert to 2D.
		real_t w2 = p_radius * p_radius - dist * dist;
		if (w2 < (real_t)CMP_EPSILON) {
			return false; // Avoid numerical error.
		}
		Size2 size(Math::sqrt(w2), p_height * 0.5f);

		Vector3 side_dir = axis_dir.cross(cylinder_axis).normalized();

		Vector2 from2D(side_dir.dot(p_from), p_from[p_cylinder_axis]);
		Vector2 to2D(side_dir.dot(p_to), p_to[p_cylinder_axis]);

		real_t min = 0, max = 1;

		int axis = -1;

		for (int i = 0; i < 2; i++) {
			real_t seg_from = from2D[i];
			real_t seg_to = to2D[i];
			real_t box_begin = -size[i];
			real_t box_end = size[i];
			real_t cmin, cmax;

			if (seg_from < seg_to) {
				if (seg_from > box_end || seg_to < box_begin) {
					return false;
				}
				real_t length = seg_to - seg_from;
				cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
				cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;

			} else {
				if (seg_to > box_end || seg_from < box_begin) {
					return false;
				}
				real_t length = seg_to - seg_from;
				cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
				cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
			}

			if (cmin > min) {
				min = cmin;
				axis = i;
			}
			if (cmax < max) {
				max = cmax;
			}
			if (max < min) {
				return false;
			}
		}

		// Convert to 3D again.
		Vector3 result = p_from + (rel * min);
		Vector3 res_normal = result;

		if (axis == 0) {
			res_normal[p_cylinder_axis] = 0;
		} else {
			int axis_side = (p_cylinder_axis + 1) % 3;
			res_normal[axis_side] = 0;
			axis_side = (axis_side + 1) % 3;
			res_normal[axis_side] = 0;
		}

		res_normal.normalize();

		if (r_res) {
			*r_res = result;
		}
		if (r_norm) {
			*r_norm = res_normal;
		}

		return true;
	}

	static bool segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Plane *p_planes, int p_plane_count, Vector3 *p_res, Vector3 *p_norm) {
		real_t min = -1e20, max = 1e20;

		Vector3 rel = p_to - p_from;
		real_t rel_l = rel.length();

		if (rel_l < (real_t)CMP_EPSILON) {
			return false;
		}

		Vector3 dir = rel / rel_l;

		int min_index = -1;

		for (int i = 0; i < p_plane_count; i++) {
			const Plane &p = p_planes[i];

			real_t den = p.normal.dot(dir);

			if (Math::abs(den) <= (real_t)CMP_EPSILON) {
				continue; // Ignore parallel plane.
			}

			real_t dist = -p.distance_to(p_from) / den;

			if (den > 0) {
				// Backwards facing plane.
				if (dist < max) {
					max = dist;
				}
			} else {
				// Front facing plane.
				if (dist > min) {
					min = dist;
					min_index = i;
				}
			}
		}

		if (max <= min || min < 0 || min > rel_l || min_index == -1) { // Exit conditions.
			return false; // No intersection.
		}

		if (p_res) {
			*p_res = p_from + dir * min;
		}
		if (p_norm) {
			*p_norm = p_planes[min_index].normal;
		}

		return true;
	}

	static inline bool point_in_projected_triangle(const Vector3 &p_point, const Vector3 &p_v1, const Vector3 &p_v2, const Vector3 &p_v3) {
		Vector3 face_n = (p_v1 - p_v3).cross(p_v1 - p_v2);

		Vector3 n1 = (p_point - p_v3).cross(p_point - p_v2);

		if (face_n.dot(n1) < 0) {
			return false;
		}

		Vector3 n2 = (p_v1 - p_v3).cross(p_v1 - p_point);

		if (face_n.dot(n2) < 0) {
			return false;
		}

		Vector3 n3 = (p_v1 - p_point).cross(p_v1 - p_v2);

		if (face_n.dot(n3) < 0) {
			return false;
		}

		return true;
	}

	static inline bool triangle_sphere_intersection_test(const Vector3 *p_triangle, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact) {
		real_t d = p_normal.dot(p_sphere_pos) - p_normal.dot(p_triangle[0]);

		if (d > p_sphere_radius || d < -p_sphere_radius) {
			// Not touching the plane of the face, return.
			return false;
		}

		Vector3 contact = p_sphere_pos - (p_normal * d);

		/** 2nd) TEST INSIDE TRIANGLE **/

		if (Geometry3D::point_in_projected_triangle(contact, p_triangle[0], p_triangle[1], p_triangle[2])) {
			r_triangle_contact = contact;
			r_sphere_contact = p_sphere_pos - p_normal * p_sphere_radius;
			//printf("solved inside triangle\n");
			return true;
		}

		/** 3rd TEST INSIDE EDGE CYLINDERS **/

		const Vector3 verts[4] = { p_triangle[0], p_triangle[1], p_triangle[2], p_triangle[0] }; // for() friendly

		for (int i = 0; i < 3; i++) {
			// Check edge cylinder.

			Vector3 n1 = verts[i] - verts[i + 1];
			Vector3 n2 = p_sphere_pos - verts[i + 1];

			///@TODO Maybe discard by range here to make the algorithm quicker.

			// Check point within cylinder radius.
			Vector3 axis = n1.cross(n2).cross(n1);
			axis.normalize();

			real_t ad = axis.dot(n2);

			if (ABS(ad) > p_sphere_radius) {
				// No chance with this edge, too far away.
				continue;
			}

			// Check point within edge capsule cylinder.
			/** 4th TEST INSIDE EDGE POINTS **/

			real_t sphere_at = n1.dot(n2);

			if (sphere_at >= 0 && sphere_at < n1.dot(n1)) {
				r_triangle_contact = p_sphere_pos - axis * (axis.dot(n2));
				r_sphere_contact = p_sphere_pos - axis * p_sphere_radius;
				// Point inside here.
				return true;
			}

			real_t r2 = p_sphere_radius * p_sphere_radius;

			if (n2.length_squared() < r2) {
				Vector3 n = (p_sphere_pos - verts[i + 1]).normalized();

				r_triangle_contact = verts[i + 1];
				r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
				return true;
			}

			if (n2.distance_squared_to(n1) < r2) {
				Vector3 n = (p_sphere_pos - verts[i]).normalized();

				r_triangle_contact = verts[i];
				r_sphere_contact = p_sphere_pos - n * p_sphere_radius;
				return true;
			}

			break; // It's pointless to continue at this point, so save some CPU cycles.
		}

		return false;
	}

	static inline Vector<Vector3> clip_polygon(const Vector<Vector3> &polygon, const Plane &p_plane) {
		enum LocationCache {
			LOC_INSIDE = 1,
			LOC_BOUNDARY = 0,
			LOC_OUTSIDE = -1
		};

		if (polygon.size() == 0) {
			return polygon;
		}

		int *location_cache = (int *)alloca(sizeof(int) * polygon.size());
		int inside_count = 0;
		int outside_count = 0;

		for (int a = 0; a < polygon.size(); a++) {
			real_t dist = p_plane.distance_to(polygon[a]);
			if (dist < (real_t)-CMP_POINT_IN_PLANE_EPSILON) {
				location_cache[a] = LOC_INSIDE;
				inside_count++;
			} else {
				if (dist > (real_t)CMP_POINT_IN_PLANE_EPSILON) {
					location_cache[a] = LOC_OUTSIDE;
					outside_count++;
				} else {
					location_cache[a] = LOC_BOUNDARY;
				}
			}
		}

		if (outside_count == 0) {
			return polygon; // No changes.
		} else if (inside_count == 0) {
			return Vector<Vector3>(); // Empty.
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

	static Vector<Vector<Face3>> separate_objects(Vector<Face3> p_array);

	// Create a "wrap" that encloses the given geometry.
	static Vector<Face3> wrap_geometry(Vector<Face3> p_array, real_t *p_error = nullptr);

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

	static MeshData build_convex_mesh(const Vector<Plane> &p_planes);
	static Vector<Plane> build_sphere_planes(real_t p_radius, int p_lats, int p_lons, Vector3::Axis p_axis = Vector3::AXIS_Z);
	static Vector<Plane> build_box_planes(const Vector3 &p_extents);
	static Vector<Plane> build_cylinder_planes(real_t p_radius, real_t p_height, int p_sides, Vector3::Axis p_axis = Vector3::AXIS_Z);
	static Vector<Plane> build_capsule_planes(real_t p_radius, real_t p_height, int p_sides, int p_lats, Vector3::Axis p_axis = Vector3::AXIS_Z);

	static Vector<Vector3> compute_convex_mesh_points(const Plane *p_planes, int p_plane_count);

#define FINDMINMAX(x0, x1, x2, min, max) \
	min = max = x0;                      \
	if (x1 < min) {                      \
		min = x1;                        \
	}                                    \
	if (x1 > max) {                      \
		max = x1;                        \
	}                                    \
	if (x2 < min) {                      \
		min = x2;                        \
	}                                    \
	if (x2 > max) {                      \
		max = x2;                        \
	}

	_FORCE_INLINE_ static bool planeBoxOverlap(Vector3 normal, float d, Vector3 maxbox) {
		int q;
		Vector3 vmin, vmax;
		for (q = 0; q <= 2; q++) {
			if (normal[q] > 0.0f) {
				vmin[q] = -maxbox[q];
				vmax[q] = maxbox[q];
			} else {
				vmin[q] = maxbox[q];
				vmax[q] = -maxbox[q];
			}
		}
		if (normal.dot(vmin) + d > 0.0f) {
			return false;
		}
		if (normal.dot(vmax) + d >= 0.0f) {
			return true;
		}

		return false;
	}

/*======================== X-tests ========================*/
#define AXISTEST_X01(a, b, fa, fb)                 \
	p0 = a * v0.y - b * v0.z;                      \
	p2 = a * v2.y - b * v2.z;                      \
	if (p0 < p2) {                                 \
		min = p0;                                  \
		max = p2;                                  \
	} else {                                       \
		min = p2;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

#define AXISTEST_X2(a, b, fa, fb)                  \
	p0 = a * v0.y - b * v0.z;                      \
	p1 = a * v1.y - b * v1.z;                      \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

/*======================== Y-tests ========================*/
#define AXISTEST_Y02(a, b, fa, fb)                 \
	p0 = -a * v0.x + b * v0.z;                     \
	p2 = -a * v2.x + b * v2.z;                     \
	if (p0 < p2) {                                 \
		min = p0;                                  \
		max = p2;                                  \
	} else {                                       \
		min = p2;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

#define AXISTEST_Y1(a, b, fa, fb)                  \
	p0 = -a * v0.x + b * v0.z;                     \
	p1 = -a * v1.x + b * v1.z;                     \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

	/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)                 \
	p1 = a * v1.x - b * v1.y;                      \
	p2 = a * v2.x - b * v2.y;                      \
	if (p2 < p1) {                                 \
		min = p2;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p2;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

#define AXISTEST_Z0(a, b, fa, fb)                  \
	p0 = a * v0.x - b * v0.y;                      \
	p1 = a * v1.x - b * v1.y;                      \
	if (p0 < p1) {                                 \
		min = p0;                                  \
		max = p1;                                  \
	} else {                                       \
		min = p1;                                  \
		max = p0;                                  \
	}                                              \
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y; \
	if (min > rad || max < -rad) {                 \
		return false;                              \
	}

	_FORCE_INLINE_ static bool triangle_box_overlap(const Vector3 &boxcenter, const Vector3 boxhalfsize, const Vector3 *triverts) {
		/*    use separating axis theorem to test overlap between triangle and box */
		/*    need to test for overlap in these directions: */
		/*    1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */
		/*       we do not even need to test these) */
		/*    2) normal of the triangle */
		/*    3) crossproduct(edge from tri, {x,y,z}-directin) */
		/*       this gives 3x3=9 more tests */
		Vector3 v0, v1, v2;
		float min, max, d, p0, p1, p2, rad, fex, fey, fez;
		Vector3 normal, e0, e1, e2;

		/* This is the fastest branch on Sun */
		/* move everything so that the boxcenter is in (0,0,0) */

		v0 = triverts[0] - boxcenter;
		v1 = triverts[1] - boxcenter;
		v2 = triverts[2] - boxcenter;

		/* compute triangle edges */
		e0 = v1 - v0; /* tri edge 0 */
		e1 = v2 - v1; /* tri edge 1 */
		e2 = v0 - v2; /* tri edge 2 */

		/* Bullet 3:  */
		/*  test the 9 tests first (this was faster) */
		fex = Math::abs(e0.x);
		fey = Math::abs(e0.y);
		fez = Math::abs(e0.z);
		AXISTEST_X01(e0.z, e0.y, fez, fey);
		AXISTEST_Y02(e0.z, e0.x, fez, fex);
		AXISTEST_Z12(e0.y, e0.x, fey, fex);

		fex = Math::abs(e1.x);
		fey = Math::abs(e1.y);
		fez = Math::abs(e1.z);
		AXISTEST_X01(e1.z, e1.y, fez, fey);
		AXISTEST_Y02(e1.z, e1.x, fez, fex);
		AXISTEST_Z0(e1.y, e1.x, fey, fex);

		fex = Math::abs(e2.x);
		fey = Math::abs(e2.y);
		fez = Math::abs(e2.z);
		AXISTEST_X2(e2.z, e2.y, fez, fey);
		AXISTEST_Y1(e2.z, e2.x, fez, fex);
		AXISTEST_Z12(e2.y, e2.x, fey, fex);

		/* Bullet 1: */
		/*  first test overlap in the {x,y,z}-directions */
		/*  find min, max of the triangle each direction, and test for overlap in */
		/*  that direction -- this is equivalent to testing a minimal AABB around */
		/*  the triangle against the AABB */

		/* test in X-direction */
		FINDMINMAX(v0.x, v1.x, v2.x, min, max);
		if (min > boxhalfsize.x || max < -boxhalfsize.x) {
			return false;
		}

		/* test in Y-direction */
		FINDMINMAX(v0.y, v1.y, v2.y, min, max);
		if (min > boxhalfsize.y || max < -boxhalfsize.y) {
			return false;
		}

		/* test in Z-direction */
		FINDMINMAX(v0.z, v1.z, v2.z, min, max);
		if (min > boxhalfsize.z || max < -boxhalfsize.z) {
			return false;
		}

		/* Bullet 2: */
		/*  test if the box intersects the plane of the triangle */
		/*  compute plane equation of triangle: normal*x+d=0 */
		normal = e0.cross(e1);
		d = -normal.dot(v0); /* plane eq: normal.x+d=0 */
		return planeBoxOverlap(normal, d, boxhalfsize); /* if true, box and triangle overlaps */
	}

	static Vector<uint32_t> generate_edf(const Vector<bool> &p_voxels, const Vector3i &p_size, bool p_negative);
	static Vector<int8_t> generate_sdf8(const Vector<uint32_t> &p_positive, const Vector<uint32_t> &p_negative);

	static Vector3 triangle_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_pos) {
		Vector3 v0 = p_b - p_a;
		Vector3 v1 = p_c - p_a;
		Vector3 v2 = p_pos - p_a;

		float d00 = v0.dot(v0);
		float d01 = v0.dot(v1);
		float d11 = v1.dot(v1);
		float d20 = v2.dot(v0);
		float d21 = v2.dot(v1);
		float denom = (d00 * d11 - d01 * d01);
		if (denom == 0) {
			return Vector3(); //invalid triangle, return empty
		}
		float v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		float u = 1.0f - v - w;
		return Vector3(u, v, w);
	}

	static Color tetrahedron_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_d, const Vector3 &p_pos) {
		Vector3 vap = p_pos - p_a;
		Vector3 vbp = p_pos - p_b;

		Vector3 vab = p_b - p_a;
		Vector3 vac = p_c - p_a;
		Vector3 vad = p_d - p_a;

		Vector3 vbc = p_c - p_b;
		Vector3 vbd = p_d - p_b;
		// ScTP computes the scalar triple product
#define STP(m_a, m_b, m_c) ((m_a).dot((m_b).cross((m_c))))
		float va6 = STP(vbp, vbd, vbc);
		float vb6 = STP(vap, vac, vad);
		float vc6 = STP(vap, vad, vab);
		float vd6 = STP(vap, vab, vac);
		float v6 = 1 / STP(vab, vac, vad);
		return Color(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
#undef STP
	}

	_FORCE_INLINE_ static Vector3 octahedron_map_decode(const Vector2 &p_uv) {
		// https://twitter.com/Stubbesaurus/status/937994790553227264
		Vector2 f = p_uv * 2.0f - Vector2(1.0f, 1.0f);
		Vector3 n = Vector3(f.x, f.y, 1.0f - Math::abs(f.x) - Math::abs(f.y));
		float t = CLAMP(-n.z, 0.0f, 1.0f);
		n.x += n.x >= 0 ? -t : t;
		n.y += n.y >= 0 ? -t : t;
		return n.normalized();
	}
};

#endif // GEOMETRY_3D_H

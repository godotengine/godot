/**************************************************************************/
/*  geometry_3d.h                                                         */
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

#include "core/math/plane.h"
#include "core/templates/local_vector.h"

struct Color;
struct Face3;

namespace Geometry3D {

void get_closest_points_between_segments(const Vector3 &p_p0, const Vector3 &p_p1, const Vector3 &p_q0, const Vector3 &p_q1, Vector3 &r_ps, Vector3 &r_qt);
real_t get_closest_distance_between_segments(const Vector3 &p_p0, const Vector3 &p_p1, const Vector3 &p_q0, const Vector3 &p_q1);

bool ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = nullptr);
bool segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2, Vector3 *r_res = nullptr);
bool segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 *r_res = nullptr, Vector3 *r_norm = nullptr);

bool segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, real_t p_height, real_t p_radius, Vector3 *r_res = nullptr, Vector3 *r_norm = nullptr, int32_t p_cylinder_axis = 2);

bool segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Plane *p_planes, int32_t p_plane_count, Vector3 *r_res, Vector3 *r_norm);

Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_segment_a, const Vector3 &p_segment_b);
Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_segment_a, const Vector3 &p_segment_b);
bool triangle_sphere_intersection_test(const Vector3 &p_triangle_a, const Vector3 &p_triangle_b, const Vector3 &p_triangle_c, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact);

#ifndef DISABLE_DEPRECATED
Vector3 get_closest_point_to_segment(const Vector3 &p_point, const Vector3 *p_segment);
Vector3 get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 *p_segment);
bool triangle_sphere_intersection_test(const Vector3 *p_triangle, const Vector3 &p_normal, const Vector3 &p_sphere_pos, real_t p_sphere_radius, Vector3 &r_triangle_contact, Vector3 &r_sphere_contact);
#endif // DISABLE_DEPRECATED

bool point_in_projected_triangle(const Vector3 &p_point, const Vector3 &p_v1, const Vector3 &p_v2, const Vector3 &p_v3);

Vector<Vector3> clip_polygon(const Vector<Vector3> &p_polygon, const Plane &p_plane);
Vector<int32_t> tetrahedralize_delaunay(const Vector<Vector3> &p_points);

Vector<Face3> wrap_geometry(const Vector<Face3> &p_array, real_t *r_error = nullptr);

struct MeshData {
	struct Face {
		Plane plane;
		LocalVector<int32_t> indices;
	};

	LocalVector<Face> faces;

	struct Edge {
		int32_t vertex_a;
		int32_t vertex_b;
		int32_t face_a;
		int32_t face_b;
	};

	LocalVector<Edge> edges;

	LocalVector<Vector3> vertices;

	void optimize_vertices();
};
MeshData build_convex_mesh(const Vector<Plane> &p_planes);

Vector<Plane> build_sphere_planes(real_t p_radius, int32_t p_lats, int32_t p_lons, Vector3::Axis p_axis = Vector3::AXIS_Z);
Vector<Plane> build_box_planes(const Vector3 &p_extents);
Vector<Plane> build_cylinder_planes(real_t p_radius, real_t p_height, int32_t p_sides, Vector3::Axis p_axis = Vector3::AXIS_Z);
Vector<Plane> build_capsule_planes(real_t p_radius, real_t p_height, int32_t p_sides, int32_t p_lats, Vector3::Axis p_axis = Vector3::AXIS_Z);

Vector<Vector3> compute_convex_mesh_points(const Plane *p_planes, int32_t p_plane_count);

bool planeBoxOverlap(Vector3 p_normal, real_t p_d, Vector3 p_max_box);

bool triangle_box_overlap(const Vector3 &p_box_center, const Vector3 p_box_half_size, const Vector3 *p_tri_verts);

Vector<uint32_t> generate_edf(const Vector<bool> &p_voxels, const Vector3i &p_size, bool p_negative);
Vector<int8_t> generate_sdf8(const Vector<uint32_t> &p_positive, const Vector<uint32_t> &p_negative);

Vector3 triangle_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_pos);

Color tetrahedron_get_barycentric_coords(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector3 &p_d, const Vector3 &p_pos);

Vector3 octahedron_map_decode(const Vector2 &p_uv);

}; //namespace Geometry3D

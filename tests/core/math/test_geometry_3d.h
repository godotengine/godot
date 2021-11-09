/*************************************************************************/
/*  test_geometry_3d.h                                                   */
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

#ifndef TEST_GEOMETRY_3D_H
#define TEST_GEOMETRY_3D_H

#include "core/math/geometry_3d.h"
#include "tests/test_macros.h"

namespace TestGeometry3D {
TEST_CASE("[Geometry3D] Closest Points Between Segments") {
	struct Case {
		Vector3 p_1, p_2, p_3, p_4;
		Vector3 got_1, got_2;
		Vector3 want_1, want_2;
		Case(){};
		Case(Vector3 p_p_1, Vector3 p_p_2, Vector3 p_p_3, Vector3 p_p_4, Vector3 p_want_1, Vector3 p_want_2) :
				p_1(p_p_1), p_2(p_p_2), p_3(p_p_3), p_4(p_p_4), want_1(p_want_1), want_2(p_want_2){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(1, -1, 1), Vector3(1, 1, -1), Vector3(-1, -2, -1), Vector3(-1, 1, 1), Vector3(1, -0.2, 0.2), Vector3(-1, -0.2, 0.2)));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Geometry3D::get_closest_points_between_segments(current_case.p_1, current_case.p_2, current_case.p_3, current_case.p_4, current_case.got_1, current_case.got_2);
		CHECK(current_case.got_1.is_equal_approx(current_case.want_1));
		CHECK(current_case.got_2.is_equal_approx(current_case.want_2));
	}
}

TEST_CASE("[Geometry3D] Closest Distance Between Segments") {
	struct Case {
		Vector3 p_1, p_2, p_3, p_4;
		float want;
		Case(){};
		Case(Vector3 p_p_1, Vector3 p_p_2, Vector3 p_p_3, Vector3 p_p_4, float p_want) :
				p_1(p_p_1), p_2(p_p_2), p_3(p_p_3), p_4(p_p_4), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(1, -2, 0), Vector3(1, 2, 0), Vector3(-1, 2, 0), Vector3(-1, -2, 0), 0.0f));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		float out = Geometry3D::get_closest_distance_between_segments(current_case.p_1, current_case.p_2, current_case.p_3, current_case.p_4);
		CHECK(out == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Build Box Planes") {
	const Vector3 extents = Vector3(5, 5, 20);
	Vector<Plane> box = Geometry3D::build_box_planes(extents);
	CHECK(box.size() == 6);
	CHECK(extents.x == box[0].d);
	CHECK(box[0].normal == Vector3(1, 0, 0));
	CHECK(extents.x == box[1].d);
	CHECK(box[1].normal == Vector3(-1, 0, 0));
	CHECK(extents.y == box[2].d);
	CHECK(box[2].normal == Vector3(0, 1, 0));
	CHECK(extents.y == box[3].d);
	CHECK(box[3].normal == Vector3(0, -1, 0));
	CHECK(extents.z == box[4].d);
	CHECK(box[4].normal == Vector3(0, 0, 1));
	CHECK(extents.z == box[5].d);
	CHECK(box[5].normal == Vector3(0, 0, -1));
}

TEST_CASE("[Geometry3D] Build Capsule Planes") {
	struct Case {
		real_t radius, height;
		int sides, lats;
		Vector3::Axis axis;
		int want_size;
		Case(){};
		Case(real_t p_radius, real_t p_height, int p_sides, int p_lats, Vector3::Axis p_axis, int p_want) :
				radius(p_radius), height(p_height), sides(p_sides), lats(p_lats), axis(p_axis), want_size(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(10, 20, 6, 10, Vector3::Axis(), 126));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector<Plane> capsule = Geometry3D::build_capsule_planes(current_case.radius, current_case.height, current_case.sides, current_case.lats, current_case.axis);
		// Should equal (p_sides * p_lats) * 2 + p_sides
		CHECK(capsule.size() == current_case.want_size);
	}
}

TEST_CASE("[Geometry3D] Build Cylinder Planes") {
	struct Case {
		real_t radius, height;
		int sides;
		Vector3::Axis axis;
		int want_size;
		Case(){};
		Case(real_t p_radius, real_t p_height, int p_sides, Vector3::Axis p_axis, int p_want) :
				radius(p_radius), height(p_height), sides(p_sides), axis(p_axis), want_size(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(3.0f, 10.0f, 10, Vector3::Axis(), 12));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector<Plane> planes = Geometry3D::build_cylinder_planes(current_case.radius, current_case.height, current_case.sides, current_case.axis);
		CHECK(planes.size() == current_case.want_size);
	}
}

TEST_CASE("[Geometry3D] Build Sphere Planes") {
	struct Case {
		real_t radius;
		int lats, lons;
		Vector3::Axis axis;
		int want_size;
		Case(){};
		Case(real_t p_radius, int p_lat, int p_lons, Vector3::Axis p_axis, int p_want) :
				radius(p_radius), lats(p_lat), lons(p_lons), axis(p_axis), want_size(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(10.0f, 10, 3, Vector3::Axis(), 63));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector<Plane> planes = Geometry3D::build_sphere_planes(current_case.radius, current_case.lats, current_case.lons, current_case.axis);
		CHECK(planes.size() == 63);
	}
}

#if false
// This test has been temporarily disabled because it's really fragile and
// breaks if calculations change very slightly. For example, it breaks when
// using doubles, and it breaks when making Plane calculations more accurate.
TEST_CASE("[Geometry3D] Build Convex Mesh") {
	struct Case {
		Vector<Plane> object;
		int want_faces, want_edges, want_vertices;
		Case(){};
		Case(Vector<Plane> p_object, int p_want_faces, int p_want_edges, int p_want_vertices) :
				object(p_object), want_faces(p_want_faces), want_edges(p_want_edges), want_vertices(p_want_vertices){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Geometry3D::build_box_planes(Vector3(5, 10, 5)), 6, 12, 8));
	tt.push_back(Case(Geometry3D::build_capsule_planes(5, 5, 20, 20, Vector3::Axis()), 820, 7603, 6243));
	tt.push_back(Case(Geometry3D::build_cylinder_planes(5, 5, 20, Vector3::Axis()), 22, 100, 80));
	tt.push_back(Case(Geometry3D::build_sphere_planes(5, 5, 20), 220, 1011, 522));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Geometry3D::MeshData mesh = Geometry3D::build_convex_mesh(current_case.object);
		CHECK(mesh.faces.size() == current_case.want_faces);
		CHECK(mesh.edges.size() == current_case.want_edges);
		CHECK(mesh.vertices.size() == current_case.want_vertices);
	}
}
#endif

TEST_CASE("[Geometry3D] Clip Polygon") {
	struct Case {
		Plane clipping_plane;
		Vector<Vector3> polygon;
		bool want;
		Case(){};
		Case(Plane p_clipping_plane, Vector<Vector3> p_polygon, bool p_want) :
				clipping_plane(p_clipping_plane), polygon(p_polygon), want(p_want){};
	};
	Vector<Case> tt;
	Vector<Plane> box_planes = Geometry3D::build_box_planes(Vector3(5, 10, 5));
	Vector<Vector3> box = Geometry3D::compute_convex_mesh_points(&box_planes[0], box_planes.size());
	tt.push_back(Case(Plane(), box, true));
	tt.push_back(Case(Plane(Vector3(0, 1, 0), Vector3(0, 3, 0)), box, false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector<Vector3> output = Geometry3D::clip_polygon(current_case.polygon, current_case.clipping_plane);
		if (current_case.want) {
			CHECK(output == current_case.polygon);
		} else {
			CHECK(output != current_case.polygon);
		}
	}
}

TEST_CASE("[Geometry3D] Compute Convex Mesh Points") {
	struct Case {
		Vector<Plane> mesh;
		Vector<Vector3> want;
		Case(){};
		Case(Vector<Plane> p_mesh, Vector<Vector3> p_want) :
				mesh(p_mesh), want(p_want){};
	};
	Vector<Case> tt;
	Vector<Vector3> cube;
	cube.push_back(Vector3(-5, -5, -5));
	cube.push_back(Vector3(5, -5, -5));
	cube.push_back(Vector3(-5, 5, -5));
	cube.push_back(Vector3(5, 5, -5));
	cube.push_back(Vector3(-5, -5, 5));
	cube.push_back(Vector3(5, -5, 5));
	cube.push_back(Vector3(-5, 5, 5));
	cube.push_back(Vector3(5, 5, 5));
	tt.push_back(Case(Geometry3D::build_box_planes(Vector3(5, 5, 5)), cube));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector<Vector3> vectors = Geometry3D::compute_convex_mesh_points(&current_case.mesh[0], current_case.mesh.size());
		CHECK(vectors == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Get Closest Point To Segment") {
	struct Case {
		Vector3 point;
		Vector<Vector3> segment;
		Vector3 want;
		Case(){};
		Case(Vector3 p_point, Vector<Vector3> p_segment, Vector3 p_want) :
				point(p_point), segment(p_segment), want(p_want){};
	};
	Vector<Case> tt;
	Vector<Vector3> test_segment;
	test_segment.push_back(Vector3(1, 1, 1));
	test_segment.push_back(Vector3(5, 5, 5));
	tt.push_back(Case(Vector3(2, 1, 4), test_segment, Vector3(2.33333, 2.33333, 2.33333)));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		Vector3 output = Geometry3D::get_closest_point_to_segment(current_case.point, &current_case.segment[0]);
		CHECK(output.is_equal_approx(current_case.want));
	}
}

TEST_CASE("[Geometry3D] Plane and Box Overlap") {
	struct Case {
		Vector3 normal, max_box;
		float d;
		bool want;
		Case(){};
		Case(Vector3 p_normal, float p_d, Vector3 p_max_box, bool p_want) :
				normal(p_normal), max_box(p_max_box), d(p_d), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(3, 4, 2), 5, Vector3(5, 5, 5), true));
	tt.push_back(Case(Vector3(0, 1, 0), -10, Vector3(5, 5, 5), false));
	tt.push_back(Case(Vector3(1, 0, 0), -6, Vector3(5, 5, 5), false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool overlap = Geometry3D::planeBoxOverlap(current_case.normal, current_case.d, current_case.max_box);
		CHECK(overlap == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Is Point in Projected Triangle") {
	struct Case {
		Vector3 point, v_1, v_2, v_3;
		bool want;
		Case(){};
		Case(Vector3 p_point, Vector3 p_v_1, Vector3 p_v_2, Vector3 p_v_3, bool p_want) :
				point(p_point), v_1(p_v_1), v_2(p_v_2), v_3(p_v_3), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(1, 1, 0), Vector3(3, 0, 0), Vector3(0, 3, 0), Vector3(-3, 0, 0), true));
	tt.push_back(Case(Vector3(5, 1, 0), Vector3(3, 0, 0), Vector3(0, 3, 0), Vector3(-3, 0, 0), false));
	tt.push_back(Case(Vector3(3, 0, 0), Vector3(3, 0, 0), Vector3(0, 3, 0), Vector3(-3, 0, 0), true));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::point_in_projected_triangle(current_case.point, current_case.v_1, current_case.v_2, current_case.v_3);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Does Ray Intersect Triangle") {
	struct Case {
		Vector3 from, direction, v_1, v_2, v_3;
		Vector3 *result;
		bool want;
		Case(){};
		Case(Vector3 p_from, Vector3 p_direction, Vector3 p_v_1, Vector3 p_v_2, Vector3 p_v_3, bool p_want) :
				from(p_from), direction(p_direction), v_1(p_v_1), v_2(p_v_2), v_3(p_v_3), result(nullptr), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(0, 1, 1), Vector3(0, 0, -10), Vector3(0, 3, 0), Vector3(-3, 0, 0), Vector3(3, 0, 0), true));
	tt.push_back(Case(Vector3(5, 10, 1), Vector3(0, 0, -10), Vector3(0, 3, 0), Vector3(-3, 0, 0), Vector3(3, 0, 0), false));
	tt.push_back(Case(Vector3(0, 1, 1), Vector3(0, 0, 10), Vector3(0, 3, 0), Vector3(-3, 0, 0), Vector3(3, 0, 0), false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::ray_intersects_triangle(current_case.from, current_case.direction, current_case.v_1, current_case.v_2, current_case.v_3, current_case.result);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Does Segment Intersect Convex") {
	struct Case {
		Vector3 from, to;
		Vector<Plane> planes;
		Vector3 *result, *normal;
		bool want;
		Case(){};
		Case(Vector3 p_from, Vector3 p_to, Vector<Plane> p_planes, bool p_want) :
				from(p_from), to(p_to), planes(p_planes), result(nullptr), normal(nullptr), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(0, 0, 0), Geometry3D::build_box_planes(Vector3(5, 5, 5)), true));
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(5, 5, 5), Geometry3D::build_box_planes(Vector3(5, 5, 5)), true));
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(6, 5, 5), Geometry3D::build_box_planes(Vector3(5, 5, 5)), false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::segment_intersects_convex(current_case.from, current_case.to, &current_case.planes[0], current_case.planes.size(), current_case.result, current_case.normal);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Segment Intersects Cylinder") {
	struct Case {
		Vector3 from, to;
		real_t height, radius;
		Vector3 *result, *normal;
		bool want;
		Case(){};
		Case(Vector3 p_from, Vector3 p_to, real_t p_height, real_t p_radius, bool p_want) :
				from(p_from), to(p_to), height(p_height), radius(p_radius), result(nullptr), normal(nullptr), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(0, 0, 0), 5, 5, true));
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(6, 6, 6), 5, 5, false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::segment_intersects_cylinder(current_case.from, current_case.to, current_case.height, current_case.radius, current_case.result, current_case.normal);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Segment Intersects Cylinder") {
	struct Case {
		Vector3 from, to, sphere_pos;
		real_t radius;
		Vector3 *result, *normal;
		bool want;
		Case(){};
		Case(Vector3 p_from, Vector3 p_to, Vector3 p_sphere_pos, real_t p_radius, bool p_want) :
				from(p_from), to(p_to), sphere_pos(p_sphere_pos), radius(p_radius), result(nullptr), normal(nullptr), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(0, 0, 0), Vector3(0, 0, 0), 5, true));
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(0, 0, 2.5), Vector3(0, 0, 0), 5, true));
	tt.push_back(Case(Vector3(10, 10, 10), Vector3(5, 5, 5), Vector3(0, 0, 0), 5, false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::segment_intersects_sphere(current_case.from, current_case.to, current_case.sphere_pos, current_case.radius, current_case.result, current_case.normal);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Segment Intersects Triangle") {
	struct Case {
		Vector3 from, to, v_1, v_2, v_3, *result;
		bool want;
		Case(){};
		Case(Vector3 p_from, Vector3 p_to, Vector3 p_v_1, Vector3 p_v_2, Vector3 p_v_3, bool p_want) :
				from(p_from), to(p_to), v_1(p_v_1), v_2(p_v_2), v_3(p_v_3), result(nullptr), want(p_want){};
	};
	Vector<Case> tt;
	tt.push_back(Case(Vector3(1, 1, 1), Vector3(-1, -1, -1), Vector3(-3, 0, 0), Vector3(0, 3, 0), Vector3(3, 0, 0), true));
	tt.push_back(Case(Vector3(1, 1, 1), Vector3(3, 0, 0), Vector3(-3, 0, 0), Vector3(0, 3, 0), Vector3(3, 0, 0), true));
	tt.push_back(Case(Vector3(1, 1, 1), Vector3(10, -1, -1), Vector3(-3, 0, 0), Vector3(0, 3, 0), Vector3(3, 0, 0), false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::segment_intersects_triangle(current_case.from, current_case.to, current_case.v_1, current_case.v_2, current_case.v_3, current_case.result);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Triangle and Box Overlap") {
	struct Case {
		Vector3 box_centre;
		Vector3 box_half_size;
		Vector3 *tri_verts;
		bool want;
		Case(){};
		Case(Vector3 p_centre, Vector3 p_half_size, Vector3 *p_verts, bool p_want) :
				box_centre(p_centre), box_half_size(p_half_size), tri_verts(p_verts), want(p_want){};
	};
	Vector<Case> tt;
	Vector3 GoodTriangle[3] = { Vector3(3, 2, 3), Vector3(2, 2, 1), Vector3(2, 1, 1) };
	tt.push_back(Case(Vector3(0, 0, 0), Vector3(5, 5, 5), GoodTriangle, true));
	Vector3 BadTriangle[3] = { Vector3(100, 100, 100), Vector3(-100, -100, -100), Vector3(10, 10, 10) };
	tt.push_back(Case(Vector3(1000, 1000, 1000), Vector3(1, 1, 1), BadTriangle, false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::triangle_box_overlap(current_case.box_centre, current_case.box_half_size, current_case.tri_verts);
		CHECK(output == current_case.want);
	}
}

TEST_CASE("[Geometry3D] Triangle and Sphere Intersect") {
	struct Case {
		Vector<Vector3> triangle;
		Vector3 normal, sphere_pos, triangle_contact, sphere_contact;
		real_t sphere_radius;
		bool want;
		Case(){};
		Case(Vector<Vector3> p_triangle, Vector3 p_normal, Vector3 p_sphere_pos, real_t p_sphere_radius, bool p_want) :
				triangle(p_triangle), normal(p_normal), sphere_pos(p_sphere_pos), triangle_contact(Vector3()), sphere_contact(Vector3()), sphere_radius(p_sphere_radius), want(p_want){};
	};
	Vector<Case> tt;
	Vector<Vector3> triangle;
	triangle.push_back(Vector3(3, 0, 0));
	triangle.push_back(Vector3(-3, 0, 0));
	triangle.push_back(Vector3(0, 3, 0));
	tt.push_back(Case(triangle, Vector3(0, -1, 0), Vector3(0, 0, 0), 5, true));
	tt.push_back(Case(triangle, Vector3(0, 1, 0), Vector3(0, 0, 0), 5, true));
	tt.push_back(Case(triangle, Vector3(0, 1, 0), Vector3(20, 0, 0), 5, false));
	for (int i = 0; i < tt.size(); ++i) {
		Case current_case = tt[i];
		bool output = Geometry3D::triangle_sphere_intersection_test(&current_case.triangle[0], current_case.normal, current_case.sphere_pos, current_case.sphere_radius, current_case.triangle_contact, current_case.sphere_contact);
		CHECK(output == current_case.want);
	}
}
} // namespace TestGeometry3D

#endif // TEST_GEOMETRY_3D_H

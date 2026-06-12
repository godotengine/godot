/**************************************************************************/
/*  test_face3.cpp                                                        */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_face3)

#include "core/math/face3.h"

namespace TestFace3 {

TEST_CASE("[Face3] Default construction") {
	Face3 face;
	CHECK_MESSAGE(
			face.vertex[0] == Vector3(),
			"Default-constructed Face3 vertex[0] should be (0,0,0).");
	CHECK_MESSAGE(
			face.vertex[1] == Vector3(),
			"Default-constructed Face3 vertex[1] should be (0,0,0).");
	CHECK_MESSAGE(
			face.vertex[2] == Vector3(),
			"Default-constructed Face3 vertex[2] should be (0,0,0).");
}

TEST_CASE("[Face3] Parameterized construction") {
	Vector3 v0(1, 2, 3);
	Vector3 v1(4, 5, 6);
	Vector3 v2(7, 8, 9);
	Face3 face(v0, v1, v2);

	CHECK_MESSAGE(
			face.vertex[0] == v0,
			"Vertex 0 should match the first constructor argument.");
	CHECK_MESSAGE(
			face.vertex[1] == v1,
			"Vertex 1 should match the second constructor argument.");
	CHECK_MESSAGE(
			face.vertex[2] == v2,
			"Vertex 2 should match the third constructor argument.");
}

TEST_CASE("[Face3] Degenerate face detection") {
	Face3 valid(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0));
	CHECK_MESSAGE(
			!valid.is_degenerate(),
			"A triangle with non-zero area should not be degenerate.");

	SUBCASE("Collinear points are degenerate") {
		Face3 collinear(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0));
		CHECK_MESSAGE(
				collinear.is_degenerate(),
				"Collinear points should be detected as degenerate.");
	}

	SUBCASE("Two identical vertices are degenerate") {
		Face3 two_same(Vector3(1, 1, 1), Vector3(1, 1, 1), Vector3(0, 0, 0));
		CHECK_MESSAGE(
				two_same.is_degenerate(),
				"A face with two identical vertices should be degenerate.");
	}

	SUBCASE("Three identical vertices are degenerate") {
		Face3 all_same(Vector3(5, 5, 5), Vector3(5, 5, 5), Vector3(5, 5, 5));
		CHECK_MESSAGE(
				all_same.is_degenerate(),
				"A face with all identical vertices should be degenerate.");
	}
}

TEST_CASE("[Face3] Area calculation") {
	Face3 unit_right(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0));
	CHECK_MESSAGE(
			unit_right.get_area() == doctest::Approx(0.5),
			"Unit right triangle area should be 0.5.");

	Face3 scaled(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));
	CHECK_MESSAGE(
			scaled.get_area() == doctest::Approx(2.0),
			"Triangle with side length 2 should have area 2.0.");

	Face3 equilateral(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0.5, Math::sqrt(3.0) / 2.0, 0));
	CHECK_MESSAGE(
			equilateral.get_area() == doctest::Approx(Math::sqrt(3.0) / 4.0),
			"Equilateral triangle with side 1 should have area sqrt(3)/4.");

	Face3 degenerate(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0));
	CHECK_MESSAGE(
			degenerate.get_area() == doctest::Approx(0.0),
			"Degenerate triangle should have zero area.");
}

TEST_CASE("[Face3] Plane extraction") {
	Face3 face(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0));

	SUBCASE("Clockwise winding") {
		Plane plane = face.get_plane(CLOCKWISE);
		CHECK_MESSAGE(
				plane.normal.is_equal_approx(Vector3(0, 0, -1)),
				"Clockwise winding of XY-plane triangle should produce -Z normal.");
	}

	SUBCASE("Counter-clockwise winding") {
		Plane plane = face.get_plane(COUNTERCLOCKWISE);
		CHECK_MESSAGE(
				plane.normal.is_equal_approx(Vector3(0, 0, 1)),
				"Counter-clockwise winding of XY-plane triangle should produce +Z normal.");
	}
}

TEST_CASE("[Face3] Ray intersection") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Ray hits face center") {
		Vector3 intersection;
		bool hit = face.intersects_ray(Vector3(0.5, 0.5, 1), Vector3(0, 0, -1), &intersection);
		CHECK_MESSAGE(hit, "Ray pointing down should hit the XY-plane triangle.");
	}

	SUBCASE("Intersection point is correct") {
		Vector3 intersection;
		face.intersects_ray(Vector3(0.5, 0.5, 1), Vector3(0, 0, -1), &intersection);
		CHECK_MESSAGE(
				intersection.is_equal_approx(Vector3(0.5, 0.5, 0)),
				"Intersection point should be the projection onto the face.");
	}

	SUBCASE("Ray misses face") {
		bool hit = face.intersects_ray(Vector3(5, 5, 1), Vector3(0, 0, -1));
		CHECK_MESSAGE(!hit, "Ray outside the triangle should not intersect.");
	}

	SUBCASE("Ray parallel to face") {
		bool hit = face.intersects_ray(Vector3(0, 0, 1), Vector3(1, 0, 0));
		CHECK_MESSAGE(!hit, "Ray parallel to the face should not intersect.");
	}

	SUBCASE("Ray from behind face") {
		bool hit = face.intersects_ray(Vector3(0.5, 0.5, -1), Vector3(0, 0, -1));
		CHECK_MESSAGE(!hit, "Ray pointing away from the face should not intersect.");
	}
}

TEST_CASE("[Face3] Segment intersection") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Segment through face") {
		Vector3 intersection;
		bool hit = face.intersects_segment(Vector3(0.5, 0.5, 1), Vector3(0.5, 0.5, -1), &intersection);
		CHECK_MESSAGE(hit, "Segment passing through the face should intersect.");
		CHECK_MESSAGE(
				intersection.is_equal_approx(Vector3(0.5, 0.5, 0)),
				"Intersection point should be on the face plane.");
	}

	SUBCASE("Segment too short") {
		bool hit = face.intersects_segment(Vector3(0.5, 0.5, 2), Vector3(0.5, 0.5, 1));
		CHECK_MESSAGE(!hit, "Segment that does not reach the face should not intersect.");
	}

	SUBCASE("Segment misses face") {
		bool hit = face.intersects_segment(Vector3(5, 5, 1), Vector3(5, 5, -1));
		CHECK_MESSAGE(!hit, "Segment passing outside the triangle should not intersect.");
	}
}

TEST_CASE("[Face3] Closest point to") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Point above face center projects onto interior") {
		Vector3 closest = face.get_closest_point_to(Vector3(0.5, 0.5, 5));
		CHECK_MESSAGE(
				closest.is_equal_approx(Vector3(0.5, 0.5, 0)),
				"Closest point should be the projection onto the face interior.");
	}

	SUBCASE("Point nearest a vertex") {
		Vector3 closest = face.get_closest_point_to(Vector3(-1, -1, 0));
		CHECK_MESSAGE(
				closest.is_equal_approx(Vector3(0, 0, 0)),
				"Closest point should be vertex (0,0,0).");
	}

	SUBCASE("Point nearest an edge") {
		Vector3 closest = face.get_closest_point_to(Vector3(3, 0, 0));
		CHECK_MESSAGE(
				closest.is_equal_approx(Vector3(2, 0, 0)),
				"Closest point should be clamped to the edge endpoint.");
	}

	SUBCASE("Point on the face") {
		Vector3 on_face(0.5, 0.5, 0);
		Vector3 closest = face.get_closest_point_to(on_face);
		CHECK_MESSAGE(
				closest.is_equal_approx(on_face),
				"Closest point to a point on the face should be the point itself.");
	}
}

TEST_CASE("[Face3] Split by plane") {
	Face3 face(Vector3(0, 0, 0), Vector3(4, 0, 0), Vector3(0, 4, 0));

	SUBCASE("Plane splits face into pieces") {
		// Plane at x=1, normal pointing +X: cuts off a small triangle on the left.
		Plane split_plane(Vector3(1, 0, 0), 1);
		Face3 results[3];
		bool is_over[3];
		int count = face.split_by_plane(split_plane, results, is_over);
		CHECK_MESSAGE(
				count >= 2,
				"Splitting a triangle by a plane through its interior should produce at least 2 faces.");
	}

	SUBCASE("Face entirely on one side") {
		// Plane at x=10: the entire face is below x=10.
		Plane far_plane(Vector3(1, 0, 0), 10);
		Face3 results[3];
		bool is_over[3];
		int count = face.split_by_plane(far_plane, results, is_over);
		CHECK_MESSAGE(
				count == 1,
				"A face entirely on one side of the plane should produce 1 result face.");
		CHECK_MESSAGE(
				!is_over[0],
				"The result face should be under the plane.");
	}

	SUBCASE("Degenerate face returns 0") {
		Face3 degen(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0));
		Plane plane(Vector3(0, 1, 0), 0);
		Face3 results[3];
		bool is_over[3];
		ERR_PRINT_OFF;
		int count = degen.split_by_plane(plane, results, is_over);
		ERR_PRINT_ON;
		CHECK_MESSAGE(
				count == 0,
				"Splitting a degenerate face should return 0.");
	}
}

TEST_CASE("[Face3] AABB intersection") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Face inside AABB") {
		AABB aabb(Vector3(-1, -1, -1), Vector3(4, 4, 4));
		CHECK_MESSAGE(
				face.intersects_aabb(aabb),
				"A face fully contained in an AABB should intersect.");
	}

	SUBCASE("Face outside AABB") {
		AABB aabb(Vector3(10, 10, 10), Vector3(1, 1, 1));
		CHECK_MESSAGE(
				!face.intersects_aabb(aabb),
				"A face far from the AABB should not intersect.");
	}

	SUBCASE("Face partially overlapping AABB") {
		AABB aabb(Vector3(1, -1, -1), Vector3(4, 4, 4));
		CHECK_MESSAGE(
				face.intersects_aabb(aabb),
				"A face partially overlapping the AABB should intersect.");
	}
}

TEST_CASE("[Face3] AABB intersection (intersects_aabb2)") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Face inside AABB") {
		AABB aabb(Vector3(-1, -1, -1), Vector3(4, 4, 4));
		CHECK_MESSAGE(
				face.intersects_aabb2(aabb),
				"A face fully contained in an AABB should intersect (aabb2).");
	}

	SUBCASE("Face outside AABB") {
		AABB aabb(Vector3(10, 10, 10), Vector3(1, 1, 1));
		CHECK_MESSAGE(
				!face.intersects_aabb2(aabb),
				"A face far from the AABB should not intersect (aabb2).");
	}
}

TEST_CASE("[Face3] project_range") {
	Face3 face(Vector3(0, 0, 0), Vector3(3, 0, 0), Vector3(0, 4, 0));

	SUBCASE("Project along X axis with identity transform") {
		real_t min_val, max_val;
		face.project_range(Vector3(1, 0, 0), Transform3D(), min_val, max_val);
		CHECK_MESSAGE(
				min_val == doctest::Approx(0.0),
				"Minimum projection on X should be 0.");
		CHECK_MESSAGE(
				max_val == doctest::Approx(3.0),
				"Maximum projection on X should be 3.");
	}

	SUBCASE("Project along Y axis with identity transform") {
		real_t min_val, max_val;
		face.project_range(Vector3(0, 1, 0), Transform3D(), min_val, max_val);
		CHECK_MESSAGE(
				min_val == doctest::Approx(0.0),
				"Minimum projection on Y should be 0.");
		CHECK_MESSAGE(
				max_val == doctest::Approx(4.0),
				"Maximum projection on Y should be 4.");
	}

	SUBCASE("Project with translation transform") {
		Transform3D translated;
		translated.origin = Vector3(10, 0, 0);
		real_t min_val, max_val;
		face.project_range(Vector3(1, 0, 0), translated, min_val, max_val);
		CHECK_MESSAGE(
				min_val == doctest::Approx(10.0),
				"Minimum projection on X with +10 translation should be 10.");
		CHECK_MESSAGE(
				max_val == doctest::Approx(13.0),
				"Maximum projection on X with +10 translation should be 13.");
	}
}

TEST_CASE("[Face3] get_support") {
	Face3 face(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0));

	SUBCASE("Face-aligned normal returns all 3 vertices") {
		Plane face_plane = face.get_plane(COUNTERCLOCKWISE);
		Vector3 vertices[3];
		int count = 0;
		face.get_support(face_plane.normal, Transform3D(), vertices, &count, 3);
		CHECK_MESSAGE(
				count == 3,
				"Support along the face normal should return all 3 vertices.");
	}

	SUBCASE("Vertex-aligned normal returns 1 vertex") {
		// Normal pointing strongly toward vertex (2,0,0).
		Vector3 vertices[3];
		int count = 0;
		face.get_support(Vector3(1, 0, 0).normalized(), Transform3D(), vertices, &count, 3);
		CHECK_MESSAGE(
				count == 1,
				"Support along a vertex-direction should return 1 vertex.");
		CHECK_MESSAGE(
				vertices[0].is_equal_approx(Vector3(2, 0, 0)),
				"The support vertex should be (2,0,0).");
	}
}

TEST_CASE("[Face3] get_aabb") {
	Face3 face(Vector3(-1, 2, 3), Vector3(4, -5, 6), Vector3(7, 8, -9));
	AABB aabb = face.get_aabb();

	CHECK_MESSAGE(
			aabb.position.is_equal_approx(Vector3(-1, -5, -9)),
			"AABB position should be the minimum of all vertex components.");
	CHECK_MESSAGE(
			aabb.has_point(face.vertex[0]),
			"AABB should contain vertex 0.");
	CHECK_MESSAGE(
			aabb.has_point(face.vertex[1]),
			"AABB should contain vertex 1.");
	CHECK_MESSAGE(
			aabb.has_point(face.vertex[2]),
			"AABB should contain vertex 2.");
}

TEST_CASE("[Face3] String conversion") {
	Face3 face(Vector3(1, 2, 3), Vector3(4, 5, 6), Vector3(7, 8, 9));
	String str = String(face);
	CHECK_MESSAGE(
			str.length() > 0,
			"String conversion should produce a non-empty string.");
}

} // namespace TestFace3

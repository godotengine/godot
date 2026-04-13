/**************************************************************************/
/*  test_canvas_occluder_cull_solid.cpp                                   */
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

TEST_FORCE_LINK(test_canvas_occluder_cull_solid)

#include "core/math/geometry_2d.h"
#include "core/math/transform_2d.h"
#include "scene/2d/light_occluder_2d.h"

namespace TestCanvasOccluderCullSolid {

TEST_SUITE("[OccluderPolygon2D][CULL_SOLID]") {
	TEST_CASE("[OccluderPolygon2D] Enum round-trip") {
		Ref<OccluderPolygon2D> occ;
		occ.instantiate();
		occ->set_cull_mode(OccluderPolygon2D::CULL_SOLID);
		CHECK(occ->get_cull_mode() == OccluderPolygon2D::CULL_SOLID);
	}

	TEST_CASE("[OccluderPolygon2D] Point-in-polygon with transform") {
		// Square polygon centered at origin: (-50,-50) to (50,50).
		Vector<Vector2> square;
		square.push_back(Vector2(-50, -50));
		square.push_back(Vector2(50, -50));
		square.push_back(Vector2(50, 50));
		square.push_back(Vector2(-50, 50));

		// Apply a transform: scale by 2 and translate by (100, 100).
		Transform2D xform;
		xform.scale_basis(Size2(2, 2));
		xform.columns[2] = Vector2(100, 100);

		// A point at world (100, 100) maps to local (0, 0) — inside the polygon.
		Vector2 world_inside(100, 100);
		Vector2 local_inside = xform.affine_inverse().xform(world_inside);
		CHECK(Geometry2D::is_point_in_polygon(local_inside, square));

		// A point at world (300, 300) maps to local (100, 100) — outside the polygon.
		Vector2 world_outside(300, 300);
		Vector2 local_outside = xform.affine_inverse().xform(world_outside);
		CHECK_FALSE(Geometry2D::is_point_in_polygon(local_outside, square));
	}

	TEST_CASE("[OccluderPolygon2D] AABB pre-check rejects points outside bounding box") {
		// Triangle polygon.
		Vector<Vector2> triangle;
		triangle.push_back(Vector2(0, 0));
		triangle.push_back(Vector2(100, 0));
		triangle.push_back(Vector2(50, 100));

		// Compute AABB.
		Rect2 aabb(triangle[0], Size2());
		for (int i = 1; i < triangle.size(); i++) {
			aabb.expand_to(triangle[i]);
		}

		// Point clearly outside the AABB.
		Vector2 outside_aabb(200, 200);
		CHECK_FALSE(aabb.has_point(outside_aabb));

		// Point inside AABB but outside the triangle (near the top-right corner of the AABB).
		Vector2 inside_aabb_outside_poly(95, 95);
		CHECK(aabb.has_point(inside_aabb_outside_poly));
		CHECK_FALSE(Geometry2D::is_point_in_polygon(inside_aabb_outside_poly, triangle));
	}

	TEST_CASE("[OccluderPolygon2D] Concave polygon") {
		// L-shaped polygon (concave).
		Vector<Vector2> l_shape;
		l_shape.push_back(Vector2(0, 0));
		l_shape.push_back(Vector2(100, 0));
		l_shape.push_back(Vector2(100, 50));
		l_shape.push_back(Vector2(50, 50));
		l_shape.push_back(Vector2(50, 100));
		l_shape.push_back(Vector2(0, 100));

		// Inside the L's bottom-left area.
		CHECK(Geometry2D::is_point_in_polygon(Vector2(25, 75), l_shape));
		// Inside the L's top-right area.
		CHECK(Geometry2D::is_point_in_polygon(Vector2(75, 25), l_shape));
		// In the concave notch — outside the polygon.
		CHECK_FALSE(Geometry2D::is_point_in_polygon(Vector2(75, 75), l_shape));
		// Completely outside.
		CHECK_FALSE(Geometry2D::is_point_in_polygon(Vector2(150, 150), l_shape));
	}
}

} // namespace TestCanvasOccluderCullSolid

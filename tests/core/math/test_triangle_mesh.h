/**************************************************************************/
/*  test_triangle_mesh.h                                                  */
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

#include "core/math/triangle_mesh.h"
#include "scene/resources/3d/primitive_meshes.h"

#include "tests/test_macros.h"

namespace TestTriangleMesh {

TEST_CASE("[SceneTree][TriangleMesh] BVH creation and intersection") {
	Ref<BoxMesh> box_mesh;
	box_mesh.instantiate();

	const Vector<Face3> faces = box_mesh->get_faces();

	Ref<TriangleMesh> triangle_mesh;
	triangle_mesh.instantiate();
	CHECK(triangle_mesh->create_from_faces(Variant(faces)));

	const Vector3 begin = Vector3(0.0, 2.0, 0.0);
	const Vector3 end = Vector3(0.0, -2.0, 0.0);

	{
		Vector3 point;
		Vector3 normal;
		int32_t *surf_index = nullptr;
		int32_t face_index = -1;
		const bool has_result = triangle_mesh->intersect_segment(begin, end, point, normal, surf_index, &face_index);
		CHECK(has_result);
		CHECK(point.is_equal_approx(Vector3(0.0, 0.5, 0.0)));
		CHECK(normal.is_equal_approx(Vector3(0.0, 1.0, 0.0)));
		CHECK(surf_index == nullptr);
		REQUIRE(face_index != -1);
		CHECK(face_index == 8);
	}

	{
		Vector3 dir = begin.direction_to(end);
		Vector3 point;
		Vector3 normal;
		int32_t *surf_index = nullptr;
		int32_t face_index = -1;
		const bool has_result = triangle_mesh->intersect_ray(begin, dir, point, normal, surf_index, &face_index);
		CHECK(has_result);
		CHECK(point.is_equal_approx(Vector3(0.0, 0.5, 0.0)));
		CHECK(normal.is_equal_approx(Vector3(0.0, 1.0, 0.0)));
		CHECK(surf_index == nullptr);
		REQUIRE(face_index != -1);
		CHECK(face_index == 8);
	}
}
} // namespace TestTriangleMesh

/**************************************************************************/
/*  test_importer_mesh.h                                                  */
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

#ifndef TEST_IMPORTER_MESH_H
#define TEST_IMPORTER_MESH_H

#include "core/math/math_funcs.h"
#include "core/variant/variant.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/surface_tool.h"

#include "tests/test_macros.h"

namespace TestImporterMesh {
TEST_CASE("[ImporterMesh] Tangents are generated for a triangle") {
	Ref<ImporterMesh> mesh = memnew(ImporterMesh);
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);
	Vector3 vertex1 = Vector3(0, 0, 0);
	Vector3 vertex2 = Vector3(1, 0, 0);
	Vector3 vertex3 = Vector3(0, 1, 0);
	st->add_vertex(vertex1);
	st->add_vertex(vertex2);
	st->add_vertex(vertex3);
	st->set_normal(Vector3(0, 0, 1));
	st->add_index(0);
	st->add_index(1);
	st->add_index(2);
	Array arrays = st->commit_to_arrays();
	mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arrays, TypedArray<Array>(), Dictionary(), Ref<Material>(), "Triangle", Mesh::ARRAY_FORMAT_TEX_UV | Mesh::ARRAY_FORMAT_NORMAL | Mesh::ARRAY_FORMAT_INDEX);
	CHECK_EQ(mesh->generate_tangents(), OK);
	CHECK((mesh->get_surface_format(0) & Mesh::ARRAY_FORMAT_TANGENT) != 0);
	PackedVector3Array tangents = mesh->get_surface_arrays(0)[Mesh::ARRAY_TANGENT];
	for (int i = 0; i < tangents.size(); ++i) {
		Vector3 tangent = tangents[i];
		CHECK(!Math::is_nan(tangent.x));
		CHECK(!Math::is_nan(tangent.y));
		CHECK(!Math::is_nan(tangent.z));
		CHECK(Math::is_equal_approx(tangent.length_squared(), 1.0f));
		Vector3 expected_tangent = Vector3(1, 0, 0);
		CHECK(Math::is_equal_approx(tangent.x, expected_tangent.x));
		CHECK(Math::is_equal_approx(tangent.y, expected_tangent.y));
		CHECK(Math::is_equal_approx(tangent.z, expected_tangent.z));
	}
}
} // namespace TestImporterMesh

#endif // TEST_IMPORTER_MESH_H

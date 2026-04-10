/**************************************************************************/
/*  test_soft_body_3d.cpp                                                 */
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

TEST_FORCE_LINK(test_soft_body_3d)

#ifndef PHYSICS_3D_DISABLED

#include "scene/3d/physics/soft_body_3d.h"
#include "scene/main/window.h"

namespace TestSoftBody3D {

TEST_CASE("[SceneTree][SoftBody3D] Mesh ownership with blend shapes") {
	SoftBody3D *soft_body = memnew(SoftBody3D);

	PackedVector3Array vertices;
	vertices.push_back(Vector3(0, 1, 0));
	vertices.push_back(Vector3(-1, 0, -1));
	vertices.push_back(Vector3(1, 0, -1));
	vertices.push_back(Vector3(0, 0, 1));

	PackedInt32Array indices = { 1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2 };

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_INDEX] = indices;

	Array blend_shape;
	blend_shape.resize(Mesh::ARRAY_MAX);
	blend_shape[Mesh::ARRAY_VERTEX] = vertices;

	TypedArray<Array> blend_shapes;
	blend_shapes.push_back(blend_shape);

	Ref<ArrayMesh> input_mesh;
	input_mesh.instantiate();
	input_mesh->add_blend_shape("Blend Shape");
	input_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays, blend_shapes);

	soft_body->set_mesh(input_mesh);

	SceneTree::get_singleton()->get_root()->add_child(soft_body);

	Ref<Mesh> output_mesh = soft_body->get_mesh();
	CHECK(output_mesh != input_mesh);
	CHECK(output_mesh->get_surface_count() == 1);

	memdelete(soft_body);
}

} // namespace TestSoftBody3D

#endif // PHYSICS_3D_DISABLED

/**************************************************************************/
/*  test_gltf_draco.h                                                     */
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

#include "test_gltf.h"

#ifdef TOOLS_ENABLED

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"

namespace TestGltf {

static MeshInstance3D *_find_mesh_instance(Node *p_node) {
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh_instance) {
		return mesh_instance;
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		mesh_instance = _find_mesh_instance(p_node->get_child(i));
		if (mesh_instance) {
			return mesh_instance;
		}
	}
	return nullptr;
}

static void _check_draco_cube_import(const String &p_path) {
	Ref<GLTFDocument> document;
	document.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	Error err = document->append_from_file(p_path, state);
	REQUIRE_MESSAGE(err == OK, "Draco-compressed GLTF import failed.");
	Node *root = document->generate_scene(state);
	REQUIRE_MESSAGE(root != nullptr, "Draco-compressed GLTF scene generation failed.");

	MeshInstance3D *mesh_instance = _find_mesh_instance(root);
	REQUIRE_MESSAGE(mesh_instance != nullptr, "Imported Draco scene has no mesh instance.");
	Ref<Mesh> mesh = mesh_instance->get_mesh();
	REQUIRE_MESSAGE(mesh.is_valid(), "Imported Draco mesh is invalid.");
	CHECK(mesh->get_surface_count() == 1);
	CHECK(mesh->surface_get_primitive_type(0) == Mesh::PRIMITIVE_TRIANGLES);

	Array arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	PackedVector3Array normals = arrays[Mesh::ARRAY_NORMAL];
	PackedVector2Array uvs = arrays[Mesh::ARRAY_TEX_UV];
	PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
	CHECK(vertices.size() == 24);
	CHECK(normals.size() == 24);
	CHECK(uvs.size() == 24);
	CHECK(indices.size() == 36);

	Ref<Material> material = mesh->surface_get_material(0);
	CHECK_MESSAGE(material.is_valid(), "Imported Draco mesh surface has no material.");
	CHECK(material->get_name() == "RedMaterial");

	memdelete(root);
}

TEST_CASE("[SceneTree][Node][Editor] Import GLB with KHR_draco_mesh_compression") {
	init("gltf_draco_compressed", "res://");
	CHECK(GLTFDocument::get_supported_gltf_extensions().has("KHR_draco_mesh_compression"));
	_check_draco_cube_import("res://draco_cube.glb");
}

TEST_CASE("[SceneTree][Node][Editor] Import separate GLTF with KHR_draco_mesh_compression") {
	init("gltf_draco_compressed", "res://");
	_check_draco_cube_import("res://draco_cube.gltf");
}

TEST_CASE("[SceneTree][Node][Editor] Reject malformed KHR_draco_mesh_compression data") {
	init("gltf_draco_compressed", "res://");

	Ref<GLTFDocument> document;
	document.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	ERR_PRINT_OFF;
	Error err = document->append_from_file("res://malformed_draco.gltf", state);
	ERR_PRINT_ON;
	CHECK(err != OK);
}

} //namespace TestGltf

#endif // TOOLS_ENABLED

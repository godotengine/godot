/**************************************************************************/
/*  test_gltf_meshopt.h                                                   */
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

static MeshInstance3D *_find_meshopt_mesh_instance(Node *p_node) {
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh_instance) {
		return mesh_instance;
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		mesh_instance = _find_meshopt_mesh_instance(p_node->get_child(i));
		if (mesh_instance) {
			return mesh_instance;
		}
	}
	return nullptr;
}

static void _check_meshopt_cube_import(const String &p_path) {
	Ref<GLTFDocument> document;
	document.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	Error err = document->append_from_file(p_path, state);
	REQUIRE_MESSAGE(err == OK, "meshopt-compressed GLTF import failed.");
	Node *root = document->generate_scene(state);
	REQUIRE_MESSAGE(root != nullptr, "meshopt-compressed GLTF scene generation failed.");

	MeshInstance3D *mesh_instance = _find_meshopt_mesh_instance(root);
	REQUIRE_MESSAGE(mesh_instance != nullptr, "Imported meshopt scene has no mesh instance.");
	Ref<Mesh> mesh = mesh_instance->get_mesh();
	REQUIRE_MESSAGE(mesh.is_valid(), "Imported meshopt mesh is invalid.");
	CHECK(mesh->get_surface_count() == 1);
	CHECK(mesh->surface_get_primitive_type(0) == Mesh::PRIMITIVE_TRIANGLES);

	Array arrays = mesh->surface_get_arrays(0);
	PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];
	CHECK(vertices.size() == 8);
	CHECK(indices.size() == 36);

	memdelete(root);
}

static void _check_meshopt_quant_cube_import(const String &p_path) {
	Ref<GLTFDocument> document;
	document.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	Error err = document->append_from_file(p_path, state);
	REQUIRE_MESSAGE(err == OK, "Quantized meshopt-compressed GLTF import failed.");
	Node *root = document->generate_scene(state);
	REQUIRE_MESSAGE(root != nullptr, "Quantized meshopt-compressed GLTF scene generation failed.");

	MeshInstance3D *mesh_instance = _find_meshopt_mesh_instance(root);
	REQUIRE_MESSAGE(mesh_instance != nullptr, "Imported quantized meshopt scene has no mesh instance.");
	Ref<Mesh> mesh = mesh_instance->get_mesh();
	REQUIRE_MESSAGE(mesh.is_valid(), "Imported quantized meshopt mesh is invalid.");
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
	for (int i = 0; i < normals.size(); i++) {
		CHECK(Math::is_equal_approx(normals[i].length(), 1.0f));
	}
	for (int i = 0; i < uvs.size(); i++) {
		CHECK(uvs[i].x >= 0.0f);
		CHECK(uvs[i].x <= 1.0f);
		CHECK(uvs[i].y >= 0.0f);
		CHECK(uvs[i].y <= 1.0f);
	}

	memdelete(root);
}

TEST_CASE("[SceneTree][Node][Editor] Import GLB with EXT_meshopt_compression") {
	init("gltf_meshopt_compressed", "res://");
	CHECK(GLTFDocument::get_supported_gltf_extensions().has("EXT_meshopt_compression"));
	_check_meshopt_cube_import("res://meshopt_cube.glb");
}

TEST_CASE("[SceneTree][Node][Editor] Import gltfpack default GLB with EXT_meshopt_compression and KHR_mesh_quantization") {
	init("gltf_meshopt_compressed", "res://");
	CHECK(GLTFDocument::get_supported_gltf_extensions().has("EXT_meshopt_compression"));
	CHECK(GLTFDocument::get_supported_gltf_extensions().has("KHR_mesh_quantization"));
	_check_meshopt_quant_cube_import("res://meshopt_quant_cube.glb");
}

TEST_CASE("[SceneTree][Node][Editor] Import separate GLTF with EXT_meshopt_compression") {
	init("gltf_meshopt_compressed", "res://");
	_check_meshopt_cube_import("res://meshopt_cube.gltf");
}

TEST_CASE("[SceneTree][Node][Editor] Reject malformed EXT_meshopt_compression data") {
	init("gltf_meshopt_compressed", "res://");

	Ref<GLTFDocument> document;
	document.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	ERR_PRINT_OFF;
	Error err = document->append_from_file("res://malformed_meshopt.gltf", state);
	ERR_PRINT_ON;
	CHECK(err != OK);
}

} //namespace TestGltf

#endif // TOOLS_ENABLED

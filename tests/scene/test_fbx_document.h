/**************************************************************************/
/*  test_fbx_document.h                                                   */
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

#ifdef UFBX_WRITE_AVAILABLE

#include "modules/fbx/fbx_document.h"
#include "modules/fbx/fbx_state.h"

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/3d/primitive_meshes.h"

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestFBXDocument {

Node *create_test_scene() {
	// Create a simple scene with a mesh
	Node3D *root = memnew(Node3D);
	root->set_name("Root");

	// Create a simple box mesh
	Ref<BoxMesh> box_mesh = memnew(BoxMesh);
	box_mesh->set_size(Vector3(2.0, 2.0, 2.0));

	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_name("Box");
	mesh_instance->set_mesh(box_mesh);
	root->add_child(mesh_instance);
	mesh_instance->set_owner(root);

	return root;
}

void test_fbx_export_binary(Node *p_node) {
	Ref<FBXDocument> fbx_document;
	fbx_document.instantiate();
	Ref<GLTFState> fbx_state;
	fbx_state.instantiate();

	// Set export format to binary (0)
	fbx_document->set_export_format(0);

	// Convert scene to FBX state
	Error err_append = fbx_document->append_from_scene(p_node, fbx_state);
	REQUIRE(err_append == OK);

	// Export to binary FBX
	String export_path = TestUtils::get_temp_path("test_export_binary.fbx");
	Error err_export = fbx_document->write_to_filesystem(fbx_state, export_path);
	CHECK(err_export == OK);

	// Verify file was created and has content
	CHECK(FileAccess::file_exists(export_path));
	Ref<FileAccess> file = FileAccess::open(export_path, FileAccess::READ);
	REQUIRE(file.is_valid());
	int64_t file_size = file->get_length();
	CHECK(file_size > 0);
	file->close();

	// Clean up
	if (FileAccess::file_exists(export_path)) {
		DirAccess::remove_absolute(export_path);
	}
}

void test_fbx_export_ascii(Node *p_node) {
	Ref<FBXDocument> fbx_document;
	fbx_document.instantiate();
	Ref<GLTFState> fbx_state;
	fbx_state.instantiate();

	// Set export format to ASCII (1)
	fbx_document->set_export_format(1);

	// Convert scene to FBX state
	Error err_append = fbx_document->append_from_scene(p_node, fbx_state);
	REQUIRE(err_append == OK);

	// Export to ASCII FBX
	String export_path = TestUtils::get_temp_path("test_export_ascii.fbx");
	Error err_export = fbx_document->write_to_filesystem(fbx_state, export_path);
	CHECK(err_export == OK);

	// Verify file was created and has content
	CHECK(FileAccess::file_exists(export_path));
	Ref<FileAccess> file = FileAccess::open(export_path, FileAccess::READ);
	REQUIRE(file.is_valid());
	int64_t file_size = file->get_length();
	CHECK(file_size > 0);

	// ASCII FBX files should start with specific header
	// Binary FBX files start with specific bytes, ASCII files start with text
	String first_line = file->get_line();
	CHECK(first_line.length() > 0);
	file->close();

	// Clean up
	if (FileAccess::file_exists(export_path)) {
		DirAccess::remove_absolute(export_path);
	}
}

void test_fbx_export_with_mesh(Node *p_node) {
	Ref<FBXDocument> fbx_document;
	fbx_document.instantiate();
	Ref<GLTFState> fbx_state;
	fbx_state.instantiate();

	// Convert scene to FBX state
	Error err_append = fbx_document->append_from_scene(p_node, fbx_state);
	REQUIRE(err_append == OK);

	// Check that meshes were processed
	Array meshes = fbx_state->get_meshes();
	CHECK(meshes.size() > 0);

	// Export to FBX
	String export_path = TestUtils::get_temp_path("test_export_mesh.fbx");
	Error err_export = fbx_document->write_to_filesystem(fbx_state, export_path);
	CHECK(err_export == OK);

	// Verify file was created
	CHECK(FileAccess::file_exists(export_path));
	Ref<FileAccess> file = FileAccess::open(export_path, FileAccess::READ);
	REQUIRE(file.is_valid());
	int64_t file_size = file->get_length();
	CHECK(file_size > 0);
	file->close();

	// Clean up
	if (FileAccess::file_exists(export_path)) {
		DirAccess::remove_absolute(export_path);
	}
}

TEST_CASE("[SceneTree][FBXDocument] Export simple scene to binary FBX") {
	Node *test_scene = create_test_scene();
	test_fbx_export_binary(test_scene);
	memdelete(test_scene);
}

TEST_CASE("[SceneTree][FBXDocument] Export simple scene to ASCII FBX") {
	Node *test_scene = create_test_scene();
	test_fbx_export_ascii(test_scene);
	memdelete(test_scene);
}

TEST_CASE("[SceneTree][FBXDocument] Export scene with mesh to FBX") {
	Node *test_scene = create_test_scene();
	test_fbx_export_with_mesh(test_scene);
	memdelete(test_scene);
}

} // namespace TestFBXDocument

#endif // UFBX_WRITE_AVAILABLE

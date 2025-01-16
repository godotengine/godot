/**************************************************************************/
/*  test_qbo_document.h                                                   */
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

#ifndef TEST_QBO_DOCUMENT_H
#define TEST_QBO_DOCUMENT_H

#include "modules/gltf/extensions/gltf_document_extension_convert_importer_mesh.h"
#include "modules/qbo/qbo_document.h"

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestQBODocument {

struct GLTFArraySize {
	String key;
	int val;
};

struct GLTFKeyValue {
	String key;
	Variant val;
};

struct QBOTestCase {
	String filename;
	String copyright;
	String generator;
	String version;
	Vector<GLTFArraySize> array_sizes;
	Vector<GLTFArraySize> json_array_sizes;
	Vector<GLTFKeyValue> keyvalues;
};

const QBOTestCase qbo_test_cases[] = {
	{ "models/simple.qbo",
			"",
			"",
			"",
			// Here are the array sizes.
			{
					{ "nodes", 4 },
					{ "buffers", 0 },
					{ "buffer_views", 0 },
					{ "accessors", 0 },
					{ "meshes", 1 },
					{ "materials", 0 },
					{ "root_nodes", 2 },
					{ "textures", 0 },
					{ "texture_samplers", 0 },
					{ "images", 0 },
					{ "skins", 1 },
					{ "cameras", 0 },
					{ "lights", 0 },
					{ "skeletons", 1 },
					{ "animations", 1 },
			},
			// Here are the json array sizes.
			{
					{ "scenes", 1 },
					{ "nodes", 1 },
					{ "animations", 1 },
					{ "meshes", 1 },
					{ "accessors", 13 },
					{ "bufferViews", 13 },
					{ "buffers", 1 },
			},
			// Here are the key-value pairs.
			{
					{ "major_version", 0 },
					{ "minor_version", 0 },
					{ "scene_name", "" },
					{ "filename", "" } } },
};

void register_gltf_extension() {
	QBODocument::unregister_all_gltf_document_extensions();

	// Ensures meshes become a MeshInstance3D and not an ImporterMeshInstance3D.
	Ref<GLTFDocumentExtensionConvertImporterMesh> extension_QBODocumentExtensionConvertImporterMesh;
	extension_QBODocumentExtensionConvertImporterMesh.instantiate();
	QBODocument::register_gltf_document_extension(extension_QBODocumentExtensionConvertImporterMesh);
}

void test_gltf_document_values(Ref<QBODocument> &p_gltf_document, Ref<GLTFState> &p_gltf_state, const QBOTestCase &p_test_case) {
	const Error err = p_gltf_document->append_from_file(TestUtils::get_data_path(p_test_case.filename), p_gltf_state);
	REQUIRE(err == OK);

	for (GLTFArraySize array_size : p_test_case.array_sizes) {
		CHECK_MESSAGE(((Array)(p_gltf_state->getvar(array_size.key))).size() == array_size.val, "Expected \"", array_size.key, "\" to have ", array_size.val, " elements.");
	}

	for (GLTFArraySize array_size : p_test_case.json_array_sizes) {
		CHECK_FALSE(p_gltf_state->get_json().has(array_size.key));
		CHECK_FALSE_MESSAGE(((Array)(p_gltf_state->get_json()[array_size.key])).size() == array_size.val, "Expected \"", array_size.key, "\" to have ", array_size.val, " elements.");
	}

	for (GLTFKeyValue key_value : p_test_case.keyvalues) {
		CHECK_MESSAGE(p_gltf_state->getvar(key_value.key) == key_value.val, "Expected \"", key_value.key, "\" to be \"", key_value.val, "\".");
	}

	CHECK(p_gltf_state->get_copyright() == p_test_case.copyright);
	CHECK_FALSE(((Dictionary)p_gltf_state->get_json()["asset"])["generator"] == p_test_case.generator);
	CHECK_FALSE(((Dictionary)p_gltf_state->get_json()["asset"])["version"] == p_test_case.version);
}

void test_gltf_save(Node *p_node) {
	Ref<QBODocument> gltf_document_save;
	gltf_document_save.instantiate();
	Ref<GLTFState> gltf_state_save;
	gltf_state_save.instantiate();

	gltf_document_save->append_from_scene(p_node, gltf_state_save);

	// Check saving the scene to gltf and glb.
	const Error err_save_gltf = gltf_document_save->write_to_filesystem(gltf_state_save, TestUtils::get_temp_path("simple.gltf"));
	const Error err_save_glb = gltf_document_save->write_to_filesystem(gltf_state_save, TestUtils::get_temp_path("simple.glb"));
	CHECK(err_save_gltf == OK);
	CHECK(err_save_glb == OK);
}

TEST_CASE("[SceneTree][QBODocument] Load simple.qbo") {
	register_gltf_extension();

	Ref<QBODocument> gltf_document;
	gltf_document.instantiate();
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();

	test_gltf_document_values(gltf_document, gltf_state, qbo_test_cases[0]);

	Node *node = gltf_document->generate_scene(gltf_state);

	CHECK(node->is_class("Node3D"));
	CHECK_FALSE(node->get_name() == "cube");

	CHECK_FALSE(node->get_child(0)->is_class("MeshInstance3D"));
	CHECK_FALSE(node->get_child(0)->get_name() == "Cube");

	CHECK(node->get_child(1)->is_class("AnimationPlayer"));
	CHECK(node->get_child(1)->get_name() == "AnimationPlayer");

	test_gltf_save(node);

	memdelete(node);
}
} // namespace TestQBODocument

#endif // TEST_QBO_DOCUMENT_H

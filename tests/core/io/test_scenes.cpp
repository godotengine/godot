/**************************************************************************/
/*  test_scenes.cpp                                                       */
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

TEST_FORCE_LINK(test_scenes)

#include "core/io/file_access.h"
#include "core/io/resource.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/property_info.h"
#include "core/os/memory.h"
#include "core/string/string_name.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"
#include "tests/test_utils.h"

namespace TestScenes {

class _TestNodeForTestingNestedScenes : public Node {
	GDCLASS(_TestNodeForTestingNestedScenes, Node);

	Ref<PackedScene> _inner_scene;

public:
	void set_inner_scene(const Ref<PackedScene> p_scene) {
		_inner_scene = p_scene;
	}

	Ref<PackedScene> get_inner_scene() const {
		return _inner_scene;
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_inner_scene", "p_scene"), &_TestNodeForTestingNestedScenes::set_inner_scene);
		ClassDB::bind_method(D_METHOD("get_inner_scene"), &_TestNodeForTestingNestedScenes::get_inner_scene);

		ADD_PROPERTY(
				PropertyInfo(Variant::OBJECT, "inner_scene", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE),
				"set_inner_scene", "get_inner_scene");
	}
};

void prepare_scene(Ref<PackedScene> scene) {
	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	scene->set_meta("other_resource", child_resource);

	scene->set_name("Hello world");
	scene->set_meta("ExampleMetadata", Vector2i(40, 80));
	scene->set_meta("string", "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks");
}

void validate_scene(Ref<PackedScene> loaded_scene) {
	CHECK_MESSAGE(
			loaded_scene->get_name() == "Hello world",
			"The loaded resource name should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_scene->get_meta("ExampleMetadata") == Vector2i(40, 80),
			"The loaded resource metadata should be equal to the expected value.");
	CHECK_MESSAGE(
			loaded_scene->get_meta("string") == "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks",
			"The loaded resource metadata should be equal to the expected value.");

	const Ref<Resource> &meta_resource = loaded_scene->get_meta("other_resource");
	CHECK_MESSAGE(
			meta_resource->get_name() == "I'm a child resource",
			"The loaded child resource name should be equal to the expected value.");
}

void validate_and_instantiate_scene(Ref<PackedScene> loaded_scene) {
	bool is_null = loaded_scene.is_null();
	CHECK_MESSAGE(!is_null, "Could not open scene file");

	if (is_null) {
		return;
	}

	validate_scene(loaded_scene);

	Node *new_node = static_cast<Node *>(loaded_scene->instantiate());
	CHECK_MESSAGE(
			new_node->get_name() == "named_node",
			"The instantiated node has invalid name.");

	const Ref<PackedScene> &inner_loaded_scene = loaded_scene->get_meta("inner_scene");
	validate_scene(inner_loaded_scene);

	Node *inner_new_node = static_cast<Node *>(inner_loaded_scene->instantiate());
	CHECK_MESSAGE(
			inner_new_node->get_name() == "inner_named_node",
			"The instantiated node has invalid name.");

	memdelete(new_node);
	memdelete(inner_new_node);
}

TEST_CASE("[Scenes] Scene Saving and loading") {
	Node *inner_node = memnew(Node);
	inner_node->set_name("inner_named_node");

	Ref<PackedScene> inner_scene = memnew(PackedScene);
	inner_scene->pack(inner_node);
	prepare_scene(inner_scene);

	Node *node = memnew(Node);
	node->set_name("named_node");

	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(node);
	prepare_scene(scene);
	scene->set_meta("inner_scene", inner_scene);

	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	scene->set_meta("other_resource", child_resource);

	const String save_path_binary = TestUtils::get_temp_path("scene.res");
	const String save_path_text = TestUtils::get_temp_path("scene.tscn");

	ResourceSaver::save(scene, save_path_binary);
	ResourceSaver::save(scene, save_path_text);

	const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
	validate_and_instantiate_scene(loaded_resource_binary);

	const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
	validate_and_instantiate_scene(loaded_resource_text);

	memdelete(node);
	memdelete(inner_node);
}

void validate_nested_scene(Ref<PackedScene> loaded_scene) {
	bool is_null = loaded_scene.is_null();
	CHECK_MESSAGE(!is_null, "Could not open scene file");

	if (is_null) {
		return;
	}

	_TestNodeForTestingNestedScenes *node_for_testing = static_cast<_TestNodeForTestingNestedScenes *>(loaded_scene->instantiate());
	Ref<PackedScene> inner_scene = node_for_testing->get_inner_scene();
	Node *inner_node = inner_scene->instantiate();

	CHECK_MESSAGE(
			inner_node->get_name() == "named_node",
			"The instantiated node has invalid name.");

	memdelete(node_for_testing);
	memdelete(inner_node);
}

// This is to test a specific bug that can happen.
TEST_CASE("[Scenes] Scene Saving and loading as property") {
	ClassDB::register_class<_TestNodeForTestingNestedScenes>();

	Node *inner_node = memnew(Node);
	const Ref<PackedScene> inner_scene = memnew(PackedScene);
	inner_node->set_name("named_node");
	inner_scene->pack(inner_node);

	_TestNodeForTestingNestedScenes *nested_scene_node = memnew(_TestNodeForTestingNestedScenes);
	nested_scene_node->set_inner_scene(inner_scene);

	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(nested_scene_node);

	const String save_path_binary = TestUtils::get_temp_path("nested_scene_test.res");
	const String save_path_text = TestUtils::get_temp_path("nested_scene_test.tscn");

	ResourceSaver::save(scene, save_path_binary);
	ResourceSaver::save(scene, save_path_text);

	memdelete(inner_node);
	memdelete(nested_scene_node);

	const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
	validate_nested_scene(loaded_resource_binary);

	const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
	validate_nested_scene(loaded_resource_text);
}

void validate_simple_scene(Ref<PackedScene> loaded_scene) {
	bool is_null = loaded_scene.is_null();
	CHECK(!is_null);

	if (is_null) {
		return;
	}

	Node *instanced_node = static_cast<Node *>(loaded_scene->instantiate());
	CHECK_MESSAGE(
			instanced_node->get_name() == "SimpleNode",
			"The instantiated node has invalid name..");

	memdelete(instanced_node);
}

TEST_CASE("[Scenes] Simple scene saving and loading") {
	Node *node = memnew(Node);
	node->set_name("SimpleNode");

	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(node);

	const String save_path_binary = TestUtils::get_temp_path("simple_scene_test.res");
	const String save_path_text = TestUtils::get_temp_path("simple_scene_test.tscn");

	ResourceSaver::save(scene, save_path_binary);
	ResourceSaver::save(scene, save_path_text);

	memdelete(node);

	const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
	validate_simple_scene(loaded_resource_binary);

	const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
	validate_simple_scene(loaded_resource_text);

	// Simple scenes should not have a [resource] tag, so let's validate it.
	Ref<FileAccess> f = FileAccess::open(save_path_text, FileAccess::READ);
	bool is_null = f.is_null();
	CHECK(!is_null);

	if (is_null) {
		return;
	}

	while (!f->eof_reached()) {
		String line;
		line = f->get_line();
		CHECK_MESSAGE(
				line != "[resource]",
				"The scene file cannot have the [resource] tag");
	}
	f->close();
}
} // namespace TestScenes

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

#include "core/io/resource.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"
#include "tests/test_utils.h"

namespace TestScenes {

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
	CHECK(!is_null);

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
} //namespace TestScenes

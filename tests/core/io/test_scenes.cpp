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
#include "core/io/missing_resource.h"
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
#include "scene/resources/resource_format_text.h"
#include "tests/test_utils.h"

namespace TestScenes {
class _TestNodeForTestingNestedScenes : public Node {
	GDCLASS(_TestNodeForTestingNestedScenes, Node);

	Ref<PackedScene> _member_inner_scene;

public:
	void set_member_inner_scene(const Ref<PackedScene> p_scene) {
		_member_inner_scene = p_scene;
	}

	Ref<PackedScene> get_member_inner_scene() const {
		return _member_inner_scene;
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_member_inner_scene", "p_scene"), &_TestNodeForTestingNestedScenes::set_member_inner_scene);
		ClassDB::bind_method(D_METHOD("get_member_inner_scene"), &_TestNodeForTestingNestedScenes::get_member_inner_scene);

		ADD_PROPERTY(
				PropertyInfo(Variant::OBJECT, "member_inner_scene", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE),
				"set_member_inner_scene", "get_member_inner_scene");
	}
};

class _CustomScene : public PackedScene {
	GDCLASS(_CustomScene, PackedScene)
	StringName _special_custom_variable;

	Ref<PackedScene> _member_inner_scene;

public:
	StringName get_special_custom_variable() const {
		return _special_custom_variable;
	}

	void set_special_custom_variable(const StringName &p_contents) {
		_special_custom_variable = p_contents;
	}

	Ref<PackedScene> get_member_inner_scene() const {
		return _member_inner_scene;
	}

	void set_member_inner_scene(const Ref<PackedScene> &p_inner_scene) {
		_member_inner_scene = p_inner_scene;
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_special_custom_variable"), &_CustomScene::get_special_custom_variable);
		ClassDB::bind_method(D_METHOD("set_special_custom_variable", "contents"), &_CustomScene::set_special_custom_variable);

		ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "special_custom_variable", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_special_custom_variable", "get_special_custom_variable");

		ClassDB::bind_method(D_METHOD("get_member_inner_scene"), &_CustomScene::get_member_inner_scene);
		ClassDB::bind_method(D_METHOD("set_member_inner_scene", "member_inner_scene"), &_CustomScene::set_member_inner_scene);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "member_inner_scene", PROPERTY_HINT_RESOURCE_TYPE, PackedScene::get_class_static(), PROPERTY_USAGE_STORAGE), "set_member_inner_scene", "get_member_inner_scene");
	}
};

void prepare_scene(const Ref<PackedScene> &scene) {
	Ref<Resource> child_resource = memnew(Resource);
	child_resource->set_name("I'm a child resource");
	scene->set_meta("other_resource", child_resource);

	scene->set_name("Hello world");
	scene->set_meta("ExampleMetadata", Vector2i(40, 80));
	scene->set_meta("string", "The\nstring\nwith\nunnecessary\nline\n\t\\\nbreaks");
}

void validate_scene(const Ref<PackedScene> &loaded_scene) {
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

	CHECK_MESSAGE(meta_resource.is_valid(), "meta resource cannot be null");

	if (meta_resource.is_valid()) {
		CHECK_MESSAGE(
				meta_resource->get_name() == "I'm a child resource",
				"The loaded child resource name should be equal to the expected value.");
	}
}

bool validate_and_instantiate_inner_scene(const Ref<PackedScene> &inner_loaded_scene, const Ref<PackedScene> &loaded_scene) {
	bool is_inner_scene_valid = inner_loaded_scene.is_valid();

	CHECK_MESSAGE(
			is_inner_scene_valid,
			vformat("The inner scene inside %s metadata cannot be null.", loaded_scene->get_path()));

	if (!is_inner_scene_valid) {
		return false;
	}

	validate_scene(inner_loaded_scene);

	Node *inner_new_node = static_cast<Node *>(inner_loaded_scene->instantiate());
	CHECK_MESSAGE(
			inner_new_node->get_name() == "inner_named_node",
			"The instantiated node has invalid name.");

	memdelete(inner_new_node);
	return true;
}

bool validate_and_instantiate_scene(const Ref<PackedScene> &loaded_scene) {
	bool is_valid = loaded_scene.is_valid();
	CHECK_MESSAGE(is_valid, "Could not open scene file");

	if (!is_valid) {
		return false;
	}

	validate_scene(loaded_scene);

	Node *new_node = static_cast<Node *>(loaded_scene->instantiate());
	CHECK_MESSAGE(
			new_node->get_name() == "named_node",
			"The instantiated node has invalid name.");

	memdelete(new_node);

	const Ref<PackedScene> &inner_loaded_scene = loaded_scene->get_meta("meta_inner_scene");
	return validate_and_instantiate_inner_scene(inner_loaded_scene, loaded_scene);
}

bool validate_nested_scene(const Ref<PackedScene> &loaded_scene) {
	bool is_valid = loaded_scene.is_valid();
	CHECK_MESSAGE(is_valid, "Could not open scene file");

	if (!is_valid) {
		return false;
	}

	_TestNodeForTestingNestedScenes *node_for_testing = static_cast<_TestNodeForTestingNestedScenes *>(loaded_scene->instantiate());
	Ref<PackedScene> member_inner_scene = node_for_testing->get_member_inner_scene();

	validate_and_instantiate_inner_scene(member_inner_scene, loaded_scene);

	memdelete(node_for_testing);
	return true;
}

bool validate_simple_scene(const Ref<PackedScene> &loaded_scene) {
	bool is_valid = loaded_scene.is_valid();
	REQUIRE(is_valid);

	Node *instanced_node = static_cast<Node *>(loaded_scene->instantiate());
	CHECK_MESSAGE(
			instanced_node->get_name() == "named_node",
			"The instantiated node has invalid name.");

	memdelete(instanced_node);
	return true;
}

bool validate_custom_scene(const Ref<_CustomScene> &loaded_scene) {
	validate_and_instantiate_scene(loaded_scene);

	Ref<PackedScene> member_inner_scene = loaded_scene->get_member_inner_scene();

	return validate_and_instantiate_inner_scene(member_inner_scene, loaded_scene);
}

Ref<PackedScene> generate_inner_scene(Node *inner_node) {
	inner_node->set_name("inner_named_node");

	const Ref<PackedScene> member_inner_scene = memnew(PackedScene);
	member_inner_scene->pack(inner_node);
	prepare_scene(member_inner_scene);

	return member_inner_scene;
}

TEST_CASE("[Scenes] Scene Saving and loading") {
	Node *inner_node = memnew(Node);
	const Ref<PackedScene> meta_inner_scene = generate_inner_scene(inner_node);

	Node *node = memnew(Node);
	node->set_name("named_node");

	Ref<PackedScene> scene = memnew(PackedScene);
	prepare_scene(scene);
	scene->set_meta("meta_inner_scene", meta_inner_scene);
	scene->pack(node);

	SUBCASE("Test Binary Scene") {
		const String save_path_binary = TestUtils::get_temp_path("scene.res");

		ResourceSaver::save(scene, save_path_binary);

		const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
		validate_and_instantiate_scene(loaded_resource_binary);
	}

	SUBCASE("Test Text Scene") {
		const String save_path_text = TestUtils::get_temp_path("scene.tscn");

		ResourceSaver::save(scene, save_path_text);

		const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
		validate_and_instantiate_scene(loaded_resource_text);
	}

	memdelete(node);
	memdelete(inner_node);
}

// This is to test a specific bug that can happen.
// More specifically https://github.com/godotengine/godot/pull/121920#pullrequestreview-4846858363
TEST_CASE("[Scenes] Scene Saving and loading as property") {
	ClassDB::register_class<_TestNodeForTestingNestedScenes>();

	Node *inner_node = memnew(Node);
	const Ref<PackedScene> member_inner_scene = generate_inner_scene(inner_node);

	_TestNodeForTestingNestedScenes *nested_scene_node = memnew(_TestNodeForTestingNestedScenes);
	prepare_scene(member_inner_scene);
	nested_scene_node->set_member_inner_scene(member_inner_scene);

	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(nested_scene_node);

	SUBCASE("Test Binary Scene") {
		const String save_path_binary = TestUtils::get_temp_path("nested_scene_test.res");

		ResourceSaver::save(scene, save_path_binary);

		const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
		validate_nested_scene(loaded_resource_binary);
	}

	SUBCASE("Test Text Scene") {
		const String save_path_text = TestUtils::get_temp_path("nested_scene_test.tscn");

		ResourceSaver::save(scene, save_path_text);

		const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
		validate_nested_scene(loaded_resource_text);
	}

	memdelete(inner_node);
	memdelete(nested_scene_node);
}

TEST_CASE("[Scenes] Simple scene saving and loading") {
	Node *node = memnew(Node);
	node->set_name("named_node");

	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(node);

	SUBCASE("Test Binary Scene") {
		const String save_path_binary = TestUtils::get_temp_path("simple_scene_test.res");

		ResourceSaver::save(scene, save_path_binary);

		const Ref<PackedScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
		validate_simple_scene(loaded_resource_binary);
	}

	SUBCASE("Test Text Scene") {
		const String save_path_text = TestUtils::get_temp_path("simple_scene_test.tscn");

		ResourceSaver::save(scene, save_path_text);

		const Ref<PackedScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
		validate_simple_scene(loaded_resource_text);

		// Simple scenes should not have a [resource] tag, so let's validate it.
		Ref<FileAccess> f = FileAccess::open(save_path_text, FileAccess::READ);
		bool is_valid = f.is_valid();
		REQUIRE(is_valid);

		while (!f->eof_reached()) {
			CHECK_MESSAGE(
					f->get_line() != "[resource]",
					"The scene file cannot have the [resource] tag");
		}

		f->close();
	}

	memdelete(node);
}

TEST_CASE("[Scenes] Custom Scene Saving and loading") {
	ClassDB::register_class<_CustomScene>();

	Node *inner_node = memnew(Node);
	const Ref<PackedScene> member_inner_scene = generate_inner_scene(inner_node);

	Node *inner_meta_node = memnew(Node);
	const Ref<PackedScene> inner_meta_scene = generate_inner_scene(inner_meta_node);

	Node *node = memnew(Node);
	node->set_name("named_node");

	const Ref<_CustomScene> scene = memnew(_CustomScene);
	scene->set_member_inner_scene(member_inner_scene);
	scene->set_special_custom_variable("Custom Variable");

	scene->set_meta("meta_inner_scene", inner_meta_scene);

	scene->pack(node);

	prepare_scene(scene);

	SUBCASE("Test Binary Scene") {
		const String save_path_binary = TestUtils::get_temp_path("custom_scene.res");

		ResourceSaver::save(scene, save_path_binary);

		const Ref<_CustomScene> &loaded_resource_binary = ResourceLoader::load(save_path_binary);
		validate_custom_scene(loaded_resource_binary);
		validate_and_instantiate_scene(loaded_resource_binary);
	}

	SUBCASE("Test Text Scene") {
		const String save_path_text = TestUtils::get_temp_path("custom_scene.tscn");

		ResourceSaver::save(scene, save_path_text);

		const Ref<_CustomScene> &loaded_resource_text = ResourceLoader::load(save_path_text);
		validate_custom_scene(loaded_resource_text);
		validate_and_instantiate_scene(loaded_resource_text);
	}

	memdelete(inner_node);
	memdelete(inner_meta_node);
	memdelete(node);
}

String get_text_header(const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return String();
	}
	while (!f->eof_reached()) {
		const String header = f->get_line().strip_edges();
		if (header.begins_with("[gd_")) {
			return header;
		}
	}
	return String();
}

// Reads the `format=N` value from the header line of a text scene/resource file. Returns -1 on failure.
int get_text_format_version(const String &p_path) {
	static const String format_prefix = "format=";
	int format_version = -1;
	const String header = get_text_header(p_path);
	const int pos = header.find(format_prefix);
	if (pos == -1) {
		return format_version;
	}
	const String rest = header.substr(pos + format_prefix.length());
	int end = 0;
	while (end < rest.length() && rest[end] >= '0' && rest[end] <= '9') {
		end++;
	}
	if (end == 0) {
		return format_version;
	}

	format_version = rest.substr(0, end).to_int();
	return format_version;
}

// Returns true if any line in the file equals p_line (or starts with it, if p_prefix is true).
bool file_has_line(const String &p_path, const String &p_line, bool p_prefix = false) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return false;
	}
	while (!f->eof_reached()) {
		const String line = f->get_line().strip_edges();
		if (p_prefix ? line.begins_with(p_line) : line == p_line) {
			return true;
		}
	}
	return false;
}

Ref<PackedScene> make_simple_scene(const String &p_node_name) {
	Node *node = memnew(Node);
	node->set_name(p_node_name);
	Ref<PackedScene> scene = memnew(PackedScene);
	scene->pack(node);
	memdelete(node);
	return scene;
}

TEST_CASE("[Scenes] Text format version") {
	SUBCASE("Scene without metadata or embedded scenes is not upgraded") {
		Ref<PackedScene> scene = make_simple_scene("SimpleNode");

		const String save_path = TestUtils::get_temp_path("format_version_no_meta.tscn");
		CHECK(ResourceSaver::save(scene, save_path) == OK);

		CHECK(get_text_format_version(save_path) == ResourceLoaderText::FORMAT_VERSION_COMPAT);
		CHECK(!file_has_line(save_path, "[resource]"));
		validate_simple_scene(ResourceLoader::load(save_path));
	}

	SUBCASE("Scene with metadata is upgraded") {
		Ref<PackedScene> scene = make_simple_scene("SimpleNode");
		scene->set_meta("ExampleMetadata", Vector2i(40, 80));

		const String save_path = TestUtils::get_temp_path("format_version_meta.tscn");
		CHECK(ResourceSaver::save(scene, save_path) == OK);

		CHECK(get_text_format_version(save_path) >= ResourceLoaderText::FORMAT_VERSION_WITH_SCENE_EMBED);
		CHECK(file_has_line(save_path, "[resource]"));

		Ref<PackedScene> loaded = ResourceLoader::load(save_path);
		bool is_null = loaded.is_null();
		CHECK(!is_null);
		if (!is_null) {
			CHECK(loaded->get_meta("ExampleMetadata") == Vector2i(40, 80));
			validate_simple_scene(loaded);
		}

		// Removing the metadata should bring the version back down.
		scene->remove_meta("ExampleMetadata");
		CHECK(ResourceSaver::save(scene, save_path) == OK);
		CHECK(get_text_format_version(save_path) == ResourceLoaderText::FORMAT_VERSION_COMPAT);
		CHECK(!file_has_line(save_path, "[resource]"));
	}

	SUBCASE("Scene with resource name is upgraded") {
		Ref<PackedScene> scene = make_simple_scene("SimpleNode");
		scene->set_name("SimpleScene");

		const String save_path = TestUtils::get_temp_path("format_version_resource_name.tscn");
		CHECK(ResourceSaver::save(scene, save_path) == OK);
		CHECK(get_text_format_version(save_path) >= ResourceLoaderText::FORMAT_VERSION_WITH_SCENE_EMBED);
		CHECK(file_has_line(save_path, "[resource]"));
	}

	SUBCASE("Missing resources metadata does not count as metadata") {
		Ref<PackedScene> scene = make_simple_scene("SimpleNode");
		scene->set_meta(META_MISSING_RESOURCES, Dictionary());

		const String save_path = TestUtils::get_temp_path("format_version_missing_meta.tscn");
		CHECK(ResourceSaver::save(scene, save_path) == OK);
		CHECK(get_text_format_version(save_path) == ResourceLoaderText::FORMAT_VERSION_COMPAT);
		CHECK(!file_has_line(save_path, "[resource]"));
		validate_simple_scene(ResourceLoader::load(save_path));

		// ...but any other metadata alongside it does.
		scene->set_meta("ExampleMetadata", Vector2i(40, 80));
		CHECK(ResourceSaver::save(scene, save_path) == OK);
		CHECK(get_text_format_version(save_path) >= ResourceLoaderText::FORMAT_VERSION_WITH_SCENE_EMBED);
		CHECK(file_has_line(save_path, "[resource]"));
	}

	SUBCASE("Scene with an embedded scene in its metadata is upgraded") {
		Ref<PackedScene> scene = make_simple_scene("named_node");
		scene->set_meta("inner_scene", make_simple_scene("inner_named_node"));

		const String save_path = TestUtils::get_temp_path("format_version_embedded_in_meta.tscn");
		CHECK(ResourceSaver::save(scene, save_path) == OK);
		CHECK(get_text_format_version(save_path) >= ResourceLoaderText::FORMAT_VERSION_WITH_SCENE_EMBED);

		Ref<PackedScene> loaded = ResourceLoader::load(save_path);
		Ref<PackedScene> loaded_inner = loaded.is_valid() ? Ref<PackedScene>(loaded->get_meta("inner_scene")) : Ref<PackedScene>();
		bool is_null = loaded_inner.is_null();
		CHECK(!is_null);
		if (!is_null) {
			Node *inner_node = loaded_inner->instantiate();
			CHECK(inner_node->get_name() == "inner_named_node");
			memdelete(inner_node);
		}
	}

	SUBCASE("Non-scene resources are not upgraded") {
		Ref<Resource> resource = memnew(Resource);
		resource->set_meta("ExampleMetadata", Vector2i(40, 80));

		const String save_path = TestUtils::get_temp_path("format_version_resource_meta.tres");
		CHECK(ResourceSaver::save(resource, save_path) == OK);
		CHECK(get_text_header(save_path).begins_with("[gd_resource"));
		CHECK(get_text_format_version(save_path) == ResourceLoaderText::FORMAT_VERSION_COMPAT);

		// A big PackedByteArray still gives the older, non-scene version.
		PackedByteArray bytes;
		bytes.resize_initialized(128);
		resource->set_meta("bytes", bytes);
		CHECK(ResourceSaver::save(resource, save_path) == OK);
		CHECK(get_text_format_version(save_path) == ResourceLoaderText::FORMAT_VERSION_HAS_NEW_VARIANTS);
	}
}

} // namespace TestScenes

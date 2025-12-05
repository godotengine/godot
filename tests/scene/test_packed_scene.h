/**************************************************************************/
/*  test_packed_scene.h                                                   */
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

#include "scene/resources/packed_scene.h"

#include "tests/test_macros.h"

// Declared in global namespace because of GDCLASS macro warning (Windows):
// "Unqualified friend declaration referring to type outside of the nearest enclosing namespace
// is a Microsoft extension; add a nested name specifier".
class _TestIntArrayProperty : public Node {
	GDCLASS(_TestIntArrayProperty, Node);

	Variant property_value;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &_TestIntArrayProperty::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &_TestIntArrayProperty::get_property);
		ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "property", PROPERTY_HINT_TYPE_STRING, vformat("%d:", Variant::INT)), "set_property", "get_property");
	}

public:
	void set_property(Variant value) { property_value = value; }
	Variant get_property() const { return property_value; }
};

class _TestNodeArrayProperty : public Node {
	GDCLASS(_TestNodeArrayProperty, Node);

	Variant property_value;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &_TestNodeArrayProperty::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &_TestNodeArrayProperty::get_property);
		ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "property", PROPERTY_HINT_TYPE_STRING, vformat("%d/%d:%s", Variant::OBJECT, PROPERTY_HINT_NODE_TYPE, "Node")), "set_property", "get_property");
	}

public:
	void set_property(Variant value) { property_value = value; }
	Variant get_property() const { return property_value; }
};

class _TestDictionaryProperty : public Node {
	GDCLASS(_TestDictionaryProperty, Node);

	Variant property_value;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &_TestDictionaryProperty::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &_TestDictionaryProperty::get_property);
		ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "property", PROPERTY_HINT_TYPE_STRING, vformat("%d:;%d:", Variant::INT, Variant::INT)), "set_property", "get_property");
	}

public:
	void set_property(Variant value) { property_value = value; }
	Variant get_property() const { return property_value; }
};

class _TestNodeDictionaryProperty : public Node {
	GDCLASS(_TestNodeDictionaryProperty, Node);

	Variant property_value;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_property", "property"), &_TestNodeDictionaryProperty::set_property);
		ClassDB::bind_method(D_METHOD("get_property"), &_TestNodeDictionaryProperty::get_property);
		ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "property", PROPERTY_HINT_TYPE_STRING, vformat("%d/%d:%s;%d/%d:%s", Variant::OBJECT, PROPERTY_HINT_NODE_TYPE, "Node", Variant::OBJECT, PROPERTY_HINT_NODE_TYPE, "Node")), "set_property", "get_property");
	}

public:
	void set_property(Variant value) { property_value = value; }
	Variant get_property() const { return property_value; }
};

namespace TestPackedScene {

TEST_CASE("[PackedScene] Pack Scene and Retrieve State") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	const Error err = packed_scene.pack(scene);
	CHECK(err == OK);

	// Retrieve the packed state.
	Ref<SceneState> state = packed_scene.get_state();
	CHECK(state.is_valid());
	CHECK(state->get_node_count() == 1);
	CHECK(state->get_node_name(0) == "TestScene");

	memdelete(scene);
}

TEST_CASE("[PackedScene] Signals Preserved when Packing Scene") {
	// Create main scene
	// root
	// `- sub_node (local)
	// `- sub_scene (instance of another scene)
	//    `- sub_scene_node (owned by sub_scene)
	Node *main_scene_root = memnew(Node);
	Node *sub_node = memnew(Node);
	Node *sub_scene_root = memnew(Node);
	Node *sub_scene_node = memnew(Node);

	main_scene_root->add_child(sub_node);
	sub_node->set_owner(main_scene_root);

	sub_scene_root->add_child(sub_scene_node);
	sub_scene_node->set_owner(sub_scene_root);

	main_scene_root->add_child(sub_scene_root);
	sub_scene_root->set_owner(main_scene_root);

	SUBCASE("Signals that should be saved") {
		int main_flags = Object::CONNECT_PERSIST;
		// sub node to a node in main scene
		sub_node->connect("ready", callable_mp(main_scene_root, &Node::is_ready), main_flags);
		// subscene root to a node in main scene
		sub_scene_root->connect("ready", callable_mp(main_scene_root, &Node::is_ready), main_flags);
		//subscene root to subscene root (connected within main scene)
		sub_scene_root->connect("ready", callable_mp(sub_scene_root, &Node::is_ready), main_flags);

		// Pack the scene.
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		const Error err = packed_scene->pack(main_scene_root);
		CHECK(err == OK);

		// Make sure the right connections are in packed scene.
		Ref<SceneState> state = packed_scene->get_state();
		CHECK_EQ(state->get_connection_count(), 3);
	}

	/*
	// FIXME: This subcase requires GH-48064 to be fixed.
	SUBCASE("Signals that should not be saved") {
		int subscene_flags = Object::CONNECT_PERSIST | Object::CONNECT_INHERITED;
		// subscene node to itself
		sub_scene_node->connect("ready", callable_mp(sub_scene_node, &Node::is_ready), subscene_flags);
		// subscene node to subscene root
		sub_scene_node->connect("ready", callable_mp(sub_scene_root, &Node::is_ready), subscene_flags);
		//subscene root to subscene root (connected within sub scene)
		sub_scene_root->connect("ready", callable_mp(sub_scene_root, &Node::is_ready), subscene_flags);

		// Pack the scene.
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		const Error err = packed_scene->pack(main_scene_root);
		CHECK(err == OK);

		// Make sure the right connections are in packed scene.
		Ref<SceneState> state = packed_scene->get_state();
		CHECK_EQ(state->get_connection_count(), 0);
	}
	*/

	memdelete(main_scene_root);
}

TEST_CASE("[PackedScene] Clear Packed Scene") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Clear the packed scene.
	packed_scene.clear();

	// Check if it has been cleared.
	Ref<SceneState> state = packed_scene.get_state();
	CHECK_FALSE(state->get_node_count() == 1);

	memdelete(scene);
}

TEST_CASE("[PackedScene] Can Instantiate Packed Scene") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Check if the packed scene can be instantiated.
	const bool can_instantiate = packed_scene.can_instantiate();
	CHECK(can_instantiate == true);

	memdelete(scene);
}

TEST_CASE("[PackedScene] Instantiate Packed Scene") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Instantiate the packed scene.
	Node *instance = packed_scene.instantiate();
	CHECK(instance != nullptr);
	CHECK(instance->get_name() == "TestScene");

	memdelete(scene);
	memdelete(instance);
}

TEST_CASE("[PackedScene] Instantiate Packed Scene With Children") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Add persisting child nodes to the scene.
	Node *child1 = memnew(Node);
	child1->set_name("Child1");
	scene->add_child(child1);
	child1->set_owner(scene);

	Node *child2 = memnew(Node);
	child2->set_name("Child2");
	scene->add_child(child2);
	child2->set_owner(scene);

	// Add non persisting child node to the scene.
	Node *child3 = memnew(Node);
	child3->set_name("Child3");
	scene->add_child(child3);

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Instantiate the packed scene.
	Node *instance = packed_scene.instantiate();
	CHECK(instance != nullptr);
	CHECK(instance->get_name() == "TestScene");

	// Validate the child nodes of the instantiated scene.
	CHECK(instance->get_child_count() == 2);
	CHECK(instance->get_child(0)->get_name() == "Child1");
	CHECK(instance->get_child(1)->get_name() == "Child2");
	CHECK(instance->get_child(0)->get_owner() == instance);
	CHECK(instance->get_child(1)->get_owner() == instance);

	memdelete(scene);
	memdelete(instance);
}

TEST_CASE("[PackedScene] Set Path") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Set a new path for the packed scene.
	const String new_path = "NewTestPath";
	packed_scene.set_path(new_path);

	// Check if the path has been set correctly.
	Ref<SceneState> state = packed_scene.get_state();
	CHECK(state.is_valid());
	CHECK(state->get_path() == new_path);

	memdelete(scene);
}

TEST_CASE("[PackedScene] Replace State") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Create another scene state to replace with.
	Ref<SceneState> new_state = memnew(SceneState);
	new_state->set_path("NewPath");

	// Replace the state.
	packed_scene.replace_state(new_state);

	// Check if the state has been replaced.
	Ref<SceneState> state = packed_scene.get_state();
	CHECK(state.is_valid());
	CHECK(state == new_state);

	memdelete(scene);
}

TEST_CASE("[PackedScene] Recreate State") {
	// Create a scene to pack.
	Node *scene = memnew(Node);
	scene->set_name("TestScene");

	// Pack the scene.
	PackedScene packed_scene;
	packed_scene.pack(scene);

	// Recreate the state.
	packed_scene.recreate_state();

	// Check if the state has been recreated.
	Ref<SceneState> state = packed_scene.get_state();
	CHECK(state.is_valid());
	CHECK(state->get_node_count() == 0); // Since the state was recreated, it should be empty.

	memdelete(scene);
}

TEST_CASE("[PackedScene] Preserve Array Types With Nil Default") {
	GDREGISTER_CLASS(_TestIntArrayProperty);

	// Create a scene to pack.
	_TestIntArrayProperty *scene = memnew(_TestIntArrayProperty);
	String property_name = "property";

	// This test is pointless if default value can't be nil.
	REQUIRE(scene->get(property_name).get_type() == Variant::NIL);

	// Set property to a typed array.
	Array array;
	StringName class_name;
	Ref<Script> script;
	array.set_typed(Variant::INT, class_name, script);
	array.push_back(0);
	array.push_back(1);
	array.push_back(2);
	scene->set(property_name, array);

	// Instantiate the scene.
	Ref<PackedScene> packed_scene;
	packed_scene.instantiate();
	packed_scene->pack(scene);
	Node *instantiated_scene = packed_scene->instantiate();

	bool get_valid;
	Array get_value = instantiated_scene->get(property_name, &get_valid);
	CHECK(get_valid);
	CHECK(get_value.get_typed_builtin() == Variant::INT);
	CHECK(get_value.size() == 3);

	memdelete(scene);
	memdelete(instantiated_scene);
}

TEST_CASE("[PackedScene] Preserve Node Array Types With Nil Default") {
	GDREGISTER_CLASS(_TestNodeArrayProperty);

	// Create a scene to pack.
	_TestNodeArrayProperty *scene = memnew(_TestNodeArrayProperty);
	String property_name = "property";

	// This test is pointless if default value can't be nil.
	REQUIRE(scene->get(property_name).get_type() == Variant::NIL);

	// Set property to a typed array.
	Array array;
	StringName class_name = "Node";
	Ref<Script> script;
	array.set_typed(Variant::OBJECT, class_name, script);
	for (int i = 0; i < 3; i++) {
		Node *child = memnew(Node);
		scene->add_child(child);
		child->set_owner(scene);
		array.push_back(child);
	}
	scene->set(property_name, array);

	// Instantiate the scene.
	Ref<PackedScene> packed_scene;
	packed_scene.instantiate();
	packed_scene->pack(scene);
	Node *instantiated_scene = packed_scene->instantiate();

	bool get_valid;
	Array get_value = instantiated_scene->get(property_name, &get_valid);
	CHECK(get_valid);
	CHECK(get_value.get_typed_builtin() == Variant::OBJECT);
	CHECK(get_value.get_typed_class_name() == class_name);
	for (int i = 0; i < 3; i++) {
		CHECK(instantiated_scene->get_child(i) == get_value[i]);
	}

	memdelete(scene);
	memdelete(instantiated_scene);
}

TEST_CASE("[PackedScene] Preserve Dictionary Types With Nil Default") {
	GDREGISTER_CLASS(_TestDictionaryProperty);

	// Create a scene to pack.
	_TestDictionaryProperty *scene = memnew(_TestDictionaryProperty);
	String property_name = "property";

	// This test is pointless if default value can't be nil.
	REQUIRE(scene->get(property_name).get_type() == Variant::NIL);

	// Set property to a typed dictionary.
	Dictionary dictionary;
	StringName key_class_name;
	Ref<Script> key_script;
	StringName value_class_name;
	Ref<Script> value_script;
	dictionary.set_typed(Variant::INT, key_class_name, key_script, Variant::INT, value_class_name, value_script);
	for (int i = 0; i < 3; i++) {
		dictionary[i] = i;
	}
	scene->set(property_name, dictionary);

	// Instantiate the scene.
	Ref<PackedScene> packed_scene;
	packed_scene.instantiate();
	packed_scene->pack(scene);
	Node *instantiated_scene = packed_scene->instantiate();

	bool get_valid;
	Dictionary get_value = instantiated_scene->get(property_name, &get_valid);
	CHECK(get_valid);
	CHECK(get_value.get_typed_key_builtin() == Variant::INT);
	CHECK(get_value.get_typed_value_builtin() == Variant::INT);

	memdelete(scene);
	memdelete(instantiated_scene);
}

TEST_CASE("[PackedScene] Preserve Node Dictionary Types With Nil Default") {
	GDREGISTER_CLASS(_TestNodeDictionaryProperty);

	// Create a scene to pack.
	_TestNodeDictionaryProperty *scene = memnew(_TestNodeDictionaryProperty);
	String property_name = "property";

	// This test is pointless if default value can't be nil.
	REQUIRE(scene->get(property_name).get_type() == Variant::NIL);

	// Set property to a typed dictionary.
	Dictionary dictionary;
	StringName key_class_name = "Node";
	Ref<Script> key_script;
	StringName value_class_name = "Node";
	Ref<Script> value_script;
	dictionary.set_typed(Variant::OBJECT, key_class_name, key_script, Variant::OBJECT, value_class_name, value_script);
	for (int i = 0; i < 3; i++) {
		Node *child = memnew(Node);
		scene->add_child(child);
		child->set_owner(scene);
		dictionary[child] = child;
	}
	scene->set(property_name, dictionary);

	// Instantiate the scene.
	Ref<PackedScene> packed_scene;
	packed_scene.instantiate();
	packed_scene->pack(scene);
	Node *instantiated_scene = packed_scene->instantiate();

	bool get_valid;
	Dictionary get_value = instantiated_scene->get(property_name, &get_valid);
	CHECK(get_valid);
	CHECK(get_value.get_typed_key_builtin() == Variant::OBJECT);
	CHECK(get_value.get_typed_key_class_name() == key_class_name);
	CHECK(get_value.get_typed_value_builtin() == Variant::OBJECT);
	CHECK(get_value.get_typed_value_class_name() == value_class_name);
	for (int i = 0; i < 3; i++) {
		Node *child = instantiated_scene->get_child(i);
		CHECK(get_value.has(child));
		CHECK(get_value[child] == Variant(child));
	}

	memdelete(scene);
	memdelete(instantiated_scene);
}

} // namespace TestPackedScene

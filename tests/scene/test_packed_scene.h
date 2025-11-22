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

TEST_CASE("[PackedScene] Connections Preserved when Packing Scene with connections on reparented nodes") {
	Node *main_scene_root = memnew(Node);
	main_scene_root->set_name("some_name"); // setting name prevents an error when "find_node" is called in pack
	Node *sub_node = memnew(Node);
	Node *sub_node_2 = memnew(Node);

	main_scene_root->add_child(sub_node);
	main_scene_root->add_child(sub_node_2);
	sub_node->set_owner(main_scene_root);
	sub_node_2->set_owner(main_scene_root);

	Callable call = Callable(sub_node_2, "is_ready");
	sub_node->connect("ready", call, Object::CONNECT_PERSIST);

	Ref<PackedScene> packed_scene;
	packed_scene.instantiate();
	const Error err = packed_scene->pack(main_scene_root);
	CHECK(err == OK);

	Ref<SceneState> state = packed_scene->get_state();
	CHECK_EQ(state->get_connection_count(), 1);

	// Take child nodes from the main scene and reparent to a node not in the tree
	Node *new_root = packed_scene->instantiate();
	Node *dangling_root = memnew(Node);

	Node *node = new_root->get_child(0);
	Node *node_2 = new_root->get_child(1);
	node->set_owner(nullptr);
	node_2->set_owner(nullptr);
	node->reparent(dangling_root);
	node_2->reparent(dangling_root);
	node->set_owner(dangling_root);
	node_2->set_owner(dangling_root);

	const Error err2 = packed_scene->pack(dangling_root);
	CHECK(err2 == OK);

	// Ensure connection is still packed in the new "dangling" scene
	Ref<SceneState> new_state = packed_scene->get_state();
	CHECK_EQ(new_state->get_connection_count(), 1);

	memdelete(new_root);
	memdelete(main_scene_root);
	memdelete(dangling_root);
}

TEST_CASE("[PackedScene] Connections Preserved when Packing Scene") {
	// Create main scene for testing all connection possibilities
	//
	// root
	// `- sub_node (local)
	// `- sub_scene (instance of another scene)
	//    `- sub_scene_node (owned by sub_scene)
	// `- sub_scene_editable (instance of another scene with editable children set)
	//    `- sub_scene_node (owned by sub_scene_editable)
	//    `- sub_scene_node_2 (owned by sub_scene_editable)
	//
	Node *main_scene_root = memnew(Node);
	Node *sub_node = memnew(Node);
	Node *sub_scene_root = memnew(Node);
	Node *sub_scene_node = memnew(Node);
	Node *sub_scene_editable_root = memnew(Node);
	Node *sub_scene_editable_node = memnew(Node);

	main_scene_root->add_child(sub_node);
	sub_node->set_owner(main_scene_root);

	// non-editable subscene
	sub_scene_root->add_child(sub_scene_node);
	sub_scene_node->set_owner(sub_scene_root);

	main_scene_root->add_child(sub_scene_root);
	main_scene_root->add_child(sub_scene_editable_root);

	// editable subscene
	sub_scene_editable_root->add_child(sub_scene_editable_node);
	main_scene_root->set_editable_instance(sub_scene_editable_root, true);
	main_scene_root->set_editable_instance(sub_scene_editable_node, true);
	sub_scene_editable_node->set_owner(sub_scene_editable_root);
	sub_scene_editable_node->set_name("editable_node");

	sub_scene_root->set_owner(main_scene_root);
	sub_scene_editable_root->set_owner(main_scene_root);
	CHECK(main_scene_root->is_editable_instance(sub_scene_editable_node) == true);

	int flags = Object::CONNECT_PERSIST;

	SUBCASE("Connections that should be saved") {
		// Sub node to a node in main scene
		sub_node->connect("ready", callable_mp(main_scene_root, &Node::is_ready), flags);
		// Subscene root to a node in main scene
		sub_scene_root->connect("ready", callable_mp(main_scene_root, &Node::is_ready), flags);
		// Subscene root to subscene root (connected within main scene)
		sub_scene_root->connect("ready", callable_mp(sub_scene_root, &Node::is_ready), flags);

		// Editable Subscene child to editable Subscene root (connected within main scene)
		Callable call = Callable(sub_scene_editable_root, "is_ready");
		sub_scene_editable_node->connect("ready", call, flags);

		// Pack the scene.
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		const Error err = packed_scene->pack(main_scene_root);
		CHECK(err == OK);

		// Make sure that all connections were saved
		Ref<SceneState> state = packed_scene->get_state();
		CHECK_EQ(state->get_connection_count(), 4);
	}

	SUBCASE("Connections that should not be saved") {
		// Subscene node to itself
		sub_scene_node->connect("ready", callable_mp(sub_scene_node, &Node::is_ready), flags);

		// Subscene node to subscene root
		sub_scene_node->connect("ready", callable_mp(sub_scene_root, &Node::is_ready), flags);

		// Subscene root to subscene root (connected within sub scene)
		Callable call = Callable(sub_scene_root, "is_ready");
		sub_scene_root->connect("ready", call, flags);
		sub_scene_root->add_connection_owner(sub_scene_root, sub_scene_root, "ready", call, false);

		// Editable Subscene child to editable Subscene root (belonging to subscene)
		Callable call2 = Callable(sub_scene_editable_root, "is_ready");
		sub_scene_editable_node->connect("ready", call2, flags);
		sub_scene_editable_node->add_connection_owner(sub_scene_editable_root, sub_scene_editable_root, "ready", call2, false);

		// Pack the scene.
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		const Error err = packed_scene->pack(main_scene_root);
		CHECK(err == OK);

		// Make sure that no connections were saved.
		Ref<SceneState> state = packed_scene->get_state();
		CHECK_EQ(state->get_connection_count(), 0);
	}

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

} // namespace TestPackedScene

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

#ifndef TEST_PACKED_SCENE_H
#define TEST_PACKED_SCENE_H

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

TEST_CASE("[PackedScene] Instantiate Packed Scene With Instantiated Subscene") {
	enum TestFlags {
		F_save_text = 1 << 0,
		F_set_dcp = 1 << 1,
		F_editable_instance = 1 << 2,
		MAX_FLAG = 1 << 3,
	} test_flags;

	// Subscenes only work with actual saved paths, so we're going to go through this several times with different behaviors.

	for (test_flags = (TestFlags)0; test_flags < MAX_FLAG; test_flags = (TestFlags)(test_flags + 1)) {
		bool save_text = (test_flags & F_save_text) != 0;
		bool set_dcp = (test_flags & F_set_dcp) != 0;
		bool editable_instance = (test_flags & F_editable_instance) != 0;

		// I'm sure there's a more C++-ish way to do this, but this works
		char casename[256];
		snprintf(casename, 256, "%s, %s, %s",
				save_text ? "Text .tscn format" : "Binary .scn format",
				set_dcp ? "With default_child_parent" : "No default_child_parent",
				editable_instance ? "With editable_instance set" : "No editable_instance");

		SUBCASE(casename) {
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

			if (set_dcp) {
				scene->set_default_child_parent(child1);
			}

			// Pack the scene.
			Ref<PackedScene> packed_scene = memnew(PackedScene);
			packed_scene->pack(scene);

			const String save_path = OS::get_singleton()->get_cache_path().path_join(save_text ? "packedscene.tscn" : "packedscene.scn");
			ResourceSaver::save(packed_scene, save_path);

			// Create an enclosing scene and take ownership of the root.
			Node *enclosing = memnew(Node);
			enclosing->set_name("EnclosingScene");
			enclosing->add_child(scene);
			scene->set_owner(enclosing);

			if (editable_instance) {
				enclosing->set_editable_instance(scene, true);
			}

			// Add a direct child of the instantiated scene.
			Node *enclosing_child = memnew(Node);
			enclosing_child->set_name("EnclosingChild");
			if (editable_instance && set_dcp) {
				// The editor will save these as children at the actual path. Mimic that here, to test the code path.
				child1->add_child(enclosing_child);
			} else {
				scene->add_child(enclosing_child);
			}
			enclosing_child->set_owner(enclosing);

			// Set the path to use for instantiation, then pack the enclosing scene
			scene->set_scene_file_path(save_path);
			PackedScene packed_enclosing;
			packed_enclosing.pack(enclosing);

			// Instantiate the enclosing scene.
			Node *instance = packed_enclosing.instantiate();
			CHECK(instance != nullptr);
			CHECK(instance->get_name() == "EnclosingScene");

			// Validate the child nodes of the instantiated scene.
			REQUIRE(instance->get_child_count() == 1);
			Node *scene_instance = instance->get_child(0);
			CHECK(scene_instance->get_name() == "TestScene");
			CHECK(scene_instance->get_owner() == instance);
			CHECK(instance->is_editable_instance(scene_instance) == editable_instance);

			if (set_dcp) {
				// When default_child_parent is set, the enclosing child is instantiated under child1
				REQUIRE(scene_instance->get_child_count() == 2);

				Node *child1_instance = scene_instance->get_child(0);
				REQUIRE(child1_instance->get_child_count() == 1);
				CHECK(child1_instance->get_child(0)->get_name() == "EnclosingChild");
				CHECK(child1_instance->get_child(0)->get_owner() == instance);
			} else {
				// When default_child_parent is unset, the enclosing child is a third sibling
				REQUIRE(scene_instance->get_child_count() == 3);
				CHECK(scene_instance->get_child(2)->get_name() == "EnclosingChild");
				CHECK(scene_instance->get_child(2)->get_owner() == instance);
			}

			// in either case, children 0 and 1 are the internal children
			CHECK(scene_instance->get_child(0)->get_name() == "Child1");
			CHECK(scene_instance->get_child(1)->get_name() == "Child2");
			CHECK(scene_instance->get_child(0)->get_owner() == scene_instance);
			CHECK(scene_instance->get_child(1)->get_owner() == scene_instance);

			memdelete(instance);
			memdelete(enclosing);
		}
	}
}

} // namespace TestPackedScene

#endif // TEST_PACKED_SCENE_H

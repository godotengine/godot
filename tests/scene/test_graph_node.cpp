/**************************************************************************/
/*  test_graph_node.cpp                                                   */
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

TEST_FORCE_LINK(test_graph_node)

#ifndef ADVANCED_GUI_DISABLED

#include "core/object/message_queue.h"
#include "scene/gui/graph_node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestGraphNode {

TEST_CASE("[GraphNode][SceneTree]") {
	SUBCASE("[GraphNode] Graph Node only child on delete should not cause error.") {
		// Setup.
		GraphNode *test_node = memnew(GraphNode);
		test_node->set_name("Graph Node");
		Control *test_child = memnew(Control);
		test_child->set_name("child");
		test_node->add_child(test_child);

		// Test.
		test_node->remove_child(test_child);
		CHECK(test_node->get_child_count(false) == 0);

		memdelete(test_child);
		memdelete(test_node);
	}

	SUBCASE("[GraphNode] Graph Node with no children should report no slots.") {
		// Setup.
		GraphNode *graph_node = memnew(GraphNode);
		graph_node->set_name("Graph Node");

		Window *root = SceneTree::get_singleton()->get_root();
		root->add_child(graph_node);

		// Test.
		MessageQueue::get_singleton()->flush();
		CHECK(graph_node->get_slot_count() == 0);

		memdelete(graph_node);
	}

	SUBCASE("[GraphNode] Graph Node with no children should report no slots, even if slot properties were defined.") {
		// Setup.
		GraphNode *graph_node = memnew(GraphNode);
		graph_node->set_name("Graph Node");

		Window *root = SceneTree::get_singleton()->get_root();
		root->add_child(graph_node);

		// Test.
		graph_node->set_slot(0, true, 0, Color(1, 0, 0, 1), true, 1, Color(0, 0, 1, 1));
		graph_node->set_slot(2, true, 0, Color(0, 1, 0, 1), true, 1, Color(1, 1, 1, 1));
		MessageQueue::get_singleton()->flush();
		CHECK(graph_node->get_slot_count() == 0);

		memdelete(graph_node);
	}

	SUBCASE("[GraphNode] Graph Node should properly count slots.") {
		// Setup.
		GraphNode *graph_node = memnew(GraphNode);
		graph_node->set_name("Graph Node");

		Window *root = SceneTree::get_singleton()->get_root();
		root->add_child(graph_node);

		// Test adding and removing children.
		// Need to flush deferred functions, as that's when slots are updated.
		int expected_slot_count = 0;
		CHECK(graph_node->get_slot_count() == expected_slot_count);

		Vector<Node *> test_children;
		for (int i = 0; i < 5; ++i) {
			Control *slot = memnew(Control);
			graph_node->add_child(slot);
			test_children.push_back(slot);
			expected_slot_count++;

			MessageQueue::get_singleton()->flush();
			CHECK(graph_node->get_slot_count() == expected_slot_count);

			if ((i & 1) == 0) {
				Node *not_a_slot = memnew(Node);
				graph_node->add_child(not_a_slot);
				test_children.push_back(not_a_slot);

				MessageQueue::get_singleton()->flush();
				CHECK(graph_node->get_slot_count() == expected_slot_count);
			}
		}

		for (Node *test_child : test_children) {
			Control *c = Object::cast_to<Control>(test_child);
			if (c) {
				expected_slot_count--;
			}
			graph_node->remove_child(test_child);
			memdelete(test_child);
			MessageQueue::get_singleton()->flush();
			CHECK(graph_node->get_slot_count() == expected_slot_count);
		}

		// Cleanup.
		memdelete(graph_node);
	}

	SUBCASE("[GraphNode] Graph Node should properly report ports for slots.") {
		// Setup.
		GraphNode *graph_node = memnew(GraphNode);
		graph_node->set_name("Graph Node");

		Window *root = SceneTree::get_singleton()->get_root();
		root->add_child(graph_node);

		Vector<Control *> test_children;
		for (int i = 0; i < 6; ++i) {
			Control *slot = memnew(Control);
			graph_node->add_child(slot);
			test_children.push_back(slot);
		}

		MessageQueue::get_singleton()->flush();

		graph_node->set_slot(0, true, 0, Color(1, 0, 0, 1), true, 1, Color(0, 0, 1, 1));
		graph_node->set_slot(1, true, 0, Color(1, 0, 0, 1), false, 1, Color(0, 0, 1, 1));
		graph_node->set_slot(2, false, 0, Color(1, 0, 0, 1), true, 1, Color(0, 0, 1, 1));
		graph_node->set_slot(3, false, 0, Color(1, 0, 0, 1), false, 1, Color(0, 0, 1, 1));
		// Slot 4 left undefined.
		graph_node->set_slot(5, true, 0, Color(1, 0, 0, 1), true, 1, Color(0, 0, 1, 1));
		graph_node->set_slot(6, true, 0, Color(1, 0, 0, 1), true, 1, Color(0, 0, 1, 1));

		// Test.
		CHECK(graph_node->get_slot_count() == 6);

		CHECK(graph_node->get_slot_input_port(0) == 0);
		CHECK(graph_node->get_slot_input_port(1) == 1);
		CHECK(graph_node->get_slot_input_port(2) == -1);
		CHECK(graph_node->get_slot_input_port(3) == -1);
		CHECK(graph_node->get_slot_input_port(4) == -1);
		CHECK(graph_node->get_slot_input_port(5) == 2);

		CHECK(graph_node->get_slot_output_port(0) == 0);
		CHECK(graph_node->get_slot_output_port(1) == -1);
		CHECK(graph_node->get_slot_output_port(2) == 1);
		CHECK(graph_node->get_slot_output_port(3) == -1);
		CHECK(graph_node->get_slot_output_port(4) == -1);
		CHECK(graph_node->get_slot_output_port(5) == 2);

		graph_node->set_slot_enabled_left(5, false);
		CHECK(graph_node->get_slot_input_port(5) == -1);
		CHECK(graph_node->get_slot_output_port(5) == 2);

		graph_node->set_slot_enabled_right(5, false);
		CHECK(graph_node->get_slot_input_port(5) == -1);
		CHECK(graph_node->get_slot_output_port(5) == -1);

		graph_node->set_slot_enabled_right(4, true);
		graph_node->set_slot_enabled_left(5, true);
		graph_node->set_slot_enabled_right(5, true);
		CHECK(graph_node->get_slot_input_port(4) == -1);
		CHECK(graph_node->get_slot_input_port(5) == 2);
		CHECK(graph_node->get_slot_output_port(4) == 2);
		CHECK(graph_node->get_slot_output_port(5) == 3);

		// Cleanup.
		for (Node *test_child : test_children) {
			graph_node->remove_child(test_child);
			memdelete(test_child);
		}

		memdelete(graph_node);
	}
}

} // namespace TestGraphNode

#endif // ADVANCED_GUI_DISABLED

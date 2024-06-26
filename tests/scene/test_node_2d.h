/**************************************************************************/
/*  test_node_2d.h                                                        */
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

#ifndef TEST_NODE_2D_H
#define TEST_NODE_2D_H

#include "scene/2d/node_2d.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestNode2D {

TEST_CASE("[SceneTree][Node2D]") {
	initcoverageDataOfPjrs(4);

	SUBCASE("[Node2D][Global Transform] Global Transform should be accessible while not in SceneTree.") { // GH-79453
		Node2D *test_node = memnew(Node2D);
		test_node->set_name("node");
		Node2D *test_child = memnew(Node2D);
		test_child->set_name("child");
		test_node->add_child(test_child);

		test_node->set_global_position(Point2(1, 1));
		CHECK_EQ(test_node->get_global_position(), Point2(1, 1));
		CHECK_EQ(test_child->get_global_position(), Point2(1, 1));
		test_node->set_global_position(Point2(2, 2));
		CHECK_EQ(test_node->get_global_position(), Point2(2, 2));
		test_node->set_global_transform(Transform2D(0, Point2(3, 3)));
		CHECK_EQ(test_node->get_global_position(), Point2(3, 3));

		memdelete(test_child);
		memdelete(test_node);
	}

	SUBCASE("[Node2D][Global Transform] Global Transform should be correct after inserting node from detached tree into SceneTree.") { // GH-86841
		Node2D *main = memnew(Node2D);
		Node2D *outer = memnew(Node2D);
		Node2D *inner = memnew(Node2D);
		SceneTree::get_singleton()->get_root()->add_child(main);

		main->set_position(Point2(100, 100));
		outer->set_position(Point2(10, 0));
		inner->set_position(Point2(0, 10));

		outer->add_child(inner);
		// `inner` is still detached.
		CHECK_EQ(inner->get_global_position(), Point2(10, 10));

		main->add_child(outer);
		// `inner` is in scene tree.
		CHECK_EQ(inner->get_global_position(), Point2(110, 110));

		main->remove_child(outer);
		// `inner` is detached again.
		CHECK_EQ(inner->get_global_position(), Point2(10, 10));

		memdelete(inner);
		memdelete(outer);
		memdelete(main);
	}

    SUBCASE("[Node2D][Global Skew] Global Skew should be correct after inserting node from detached tree into SceneTree.") {
        init_coverage_funcs_set_global_skew_scale("set_global_skew", 2);
        Node2D *parent_node = memnew(Node2D);
        Node2D *child_node = memnew(Node2D);
        parent_node->add_child(child_node);

        parent_node->set_global_skew(Math_PI / 2);
        real_t parent_skew = parent_node->get_global_skew();
        CHECK_EQ(parent_skew, Math_PI / 4);
        child_node->set_global_skew(Math_PI / 4);
        real_t child_skew = child_node->get_global_skew();
        CHECK_EQ(child_skew, Math_PI / 2);

        memdelete(child_node);
        memdelete(parent_node);
        print_coverage_funcs_set_global_skew_scale();
    }

    SUBCASE("[Node2D][Global Scale] Global Scale should be correct after inserting node from detached tree into SceneTree.") {
        init_coverage_funcs_set_global_skew_scale("set_global_scale", 2);
        Node2D *parent_node = memnew(Node2D);
        Node2D *child_node = memnew(Node2D);
        parent_node->add_child(child_node);

        parent_node->set_global_scale(Size2(100, 100));
        Size2 parent_scale = parent_node->get_global_scale();
        CHECK_EQ(parent_scale, Size2(50, 50));

        child_node->set_global_scale(Size2(50, 50));
        Size2 child_scale = child_node->get_global_scale();
        CHECK_EQ(child_scale, Size2(100, 100));

        memdelete(child_node);
        memdelete(parent_node);
        print_coverage_funcs_set_global_skew_scale();
    }

    SUBCASE("[Node2D] Test set_global_rotation") {
        initializeCoverageDataSetGlobalRotation(2);

        Node2D *parent_node = memnew(Node2D);
        Node2D *child_node = memnew(Node2D);

        // Add child node to parent
        parent_node->add_child(child_node);

        // Set parent node's rotation
        parent_node->set_rotation(Math_PI / 4); // 45 degrees

        // Set child's global rotation
        child_node->set_global_rotation(Math_PI / 2); // 90 degrees


        parent_node->set_global_rotation(Math_PI / 2);

        // Get child's global rotation and verify
        real_t child_global_rotation = child_node->get_global_rotation();
        CHECK_EQ(child_global_rotation, Math_PI / 2); // Should be 90 degrees

        // Verify child's local rotation (should account for parent's rotation)
        real_t child_local_rotation = child_node->get_rotation();
        CHECK_EQ(child_local_rotation, Math_PI / 4); // Should be 45 degrees to account for parent's 45 degrees

        // Cleanup
        memdelete(child_node);
        memdelete(parent_node);
        writeCoverageDataSetGlobalRotation();
    }

    SUBCASE("[Node2D] Test Move_x") {
        initializeCoverageDataMoveX(2);
        Node2D *node = memnew(Node2D);
        node->move_x(10, false);

        // Check the new position
        CHECK(node->get_position() == Vector2(10, 0));

        // Move the node along the x-axis with scaling
        node->move_x(5, true);

        // The movement vector should be scaled according to the node's current transform
        // Since we haven't scaled the node, the movement should be direct addition
        // Let's verify the current position assuming no scaling effect
        CHECK(node->get_position() == Vector2(15, 0));
        memdelete(node);
        writeCoverageDataMoveX();
    }

	SUBCASE("[Node2D][get_relative_transform_to_parent] When p_parent == this") {
		Node2D *node = memnew(Node2D);
		CHECK_EQ(node->get_relative_transform_to_parent(node), Transform2D());
		memdelete(node);
	}

	SUBCASE("[Node2D][get_relative_transform_to_parent] When p_parent == parent_2d") {
		Node2D *parent = memnew(Node2D);
		Node2D *child = memnew(Node2D);
		parent->add_child(child);
		child->set_transform(Transform2D(-1, 0, 0, 1, 0, 0));	//The Transform2D that will flip something along the X axis.

		CHECK_EQ(child->get_relative_transform_to_parent(parent), Transform2D(-1, 0, 0, 1, 0, 0));

		memdelete(child);
		memdelete(parent);
	}

	SUBCASE("[Node2D][get_relative_transform_to_parent] Else") {
		Node2D *grandparent = memnew(Node2D);
		Node2D *parent = memnew(Node2D);
		Node2D *child = memnew(Node2D);

		grandparent->add_child(parent);
		parent->add_child(child);

		grandparent->set_transform(Transform2D(1, 0, 0, 1, 3, 3));	//No translation, no rotation, offset 3, 3
		parent->set_transform(Transform2D(1, 0, 0, 1, 2, 2));	//No translation, no rotation, offset 2, 2
		child->set_transform(Transform2D(1, 0, 0, 1, 1, 1));	//No translation, no rotation, offset 1, 1

		CHECK_EQ(child->get_relative_transform_to_parent(grandparent), Transform2D(1, 0, 0, 1, 6, 6));	//Relative transform 6, 6

		memdelete(child);
		memdelete(parent);
		memdelete(grandparent);
	}

	outputCoverageDataOfPjrs();
}

} // namespace TestNode2D

#endif // TEST_NODE_2D_H

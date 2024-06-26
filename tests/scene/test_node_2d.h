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
  SUBCASE("[Node2D] Test Move_x"){
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



}






} // namespace TestNode2D

#endif // TEST_NODE_2D_H

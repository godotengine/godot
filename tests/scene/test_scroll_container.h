/**************************************************************************/
/*  test_scroll_container.h                                               */
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

#ifndef TEST_SCROLL_CONTAINER_H
#define TEST_SCROLL_CONTAINER_H

#include "scene/gui/scroll_container.h"

#include "tests/test_macros.h"

namespace TestScrollContainer {

TEST_CASE("[SceneTree][ScrollContainer] Follow focus") {
	ScrollContainer *root = memnew(ScrollContainer);
	root->set_custom_minimum_size(Size2(100, 100));
	SceneTree::get_singleton()->get_root()->add_child(root);

	Control *ctrl = memnew(Control);
	ctrl->set_custom_minimum_size(Size2(1000, 1000));
	root->add_child(ctrl);

	Vector<Control *> children;

	for (int i = 0; i < 9; i++) {
		Control *child = memnew(Control);
		child->set_focus_mode(Control::FOCUS_ALL);
		child->set_custom_minimum_size(Size2(150, 10));
		child->set_position(Point2(150 + (i % 3) * 300, 150 + (i / 3) * 300));
		ctrl->add_child(child);
		children.push_back(child);
	}

	SceneTree::get_singleton()->process(0.1f); // Wait one frame for range configuration to complete.

	root->set_h_scroll(460);
	root->set_v_scroll(400);

	SceneTree::get_singleton()->process(0.1f); // Wait for the scroll transform to be applied.

	REQUIRE_UNARY_FALSE(root->is_following_focus());
	REQUIRE_EQ(root->get_h_scroll(), 460);
	REQUIRE_EQ(root->get_v_scroll(), 400);

	SUBCASE("No scrolling when focused by default") {
		Control *child = children[0];
		REQUIRE_UNARY_FALSE(child->has_focus());
		child->grab_focus();
		REQUIRE_UNARY(child->has_focus());

		SceneTree::get_singleton()->process(0.1f);
		CHECK_EQ(root->get_h_scroll(), 460);
		CHECK_EQ(root->get_v_scroll(), 400);
	}

	SUBCASE("Enable follow focus") {
		root->set_follow_focus(true);
		REQUIRE_UNARY(root->is_following_focus());

		SUBCASE("Scroll when focused, the control is in the upper left corner") {
			Control *child = children[0];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 208);
			CHECK_EQ(root->get_v_scroll(), 150);

			SUBCASE("Not scroll when already focused") {
				root->set_h_scroll(460);
				root->set_v_scroll(400);

				SceneTree::get_singleton()->process(0.1f);

				REQUIRE_EQ(root->get_h_scroll(), 460);
				REQUIRE_EQ(root->get_v_scroll(), 400);

				REQUIRE_UNARY(child->has_focus());
				child->grab_focus();
				REQUIRE_UNARY(child->has_focus());

				SceneTree::get_singleton()->process(0.1f);
				CHECK_EQ(root->get_h_scroll(), 460);
				CHECK_EQ(root->get_v_scroll(), 400);
			}
		}
		SUBCASE("Scroll when focused, the control is directly above") {
			Control *child = children[1];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			CHECK_EQ(root->get_h_scroll(), 460);
			CHECK_EQ(root->get_v_scroll(), 150);
		}
		SUBCASE("Scroll when focused, the control is in the upper right corner") {
			Control *child = children[2];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 750);
			CHECK_EQ(root->get_v_scroll(), 150);
		}
		SUBCASE("Scroll when focused, the control is on the left") {
			Control *child = children[3];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 208);
			CHECK_EQ(root->get_v_scroll(), 400);
		}
		SUBCASE("Not scroll when focused, since the visible area cannot be made larger") {
			Control *child = children[4];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 460);
			CHECK_EQ(root->get_v_scroll(), 400);
		}
		SUBCASE("Scroll when focused, the control is on the right") {
			Control *child = children[5];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 750);
			CHECK_EQ(root->get_v_scroll(), 400);
		}
		SUBCASE("Scroll when focused, the control is in the lower left corner") {
			Control *child = children[6];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 208);
			CHECK_EQ(root->get_v_scroll(), 668);
		}
		SUBCASE("Scroll when focused, the control is directly below") {
			Control *child = children[7];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 460);
			CHECK_EQ(root->get_v_scroll(), 668);
		}
		SUBCASE("Scroll when focused, the control is in the lower right corner") {
			Control *child = children[8];
			REQUIRE_UNARY_FALSE(child->has_focus());
			child->grab_focus();
			REQUIRE_UNARY(child->has_focus());

			SceneTree::get_singleton()->process(0.1f);
			CHECK_EQ(root->get_h_scroll(), 750);
			CHECK_EQ(root->get_v_scroll(), 668);
		}
	}

	root->queue_free();
}

} // namespace TestScrollContainer

#endif // TEST_SCROLL_CONTAINER_H

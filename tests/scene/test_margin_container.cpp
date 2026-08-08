/**************************************************************************/
/*  test_margin_container.cpp                                             */
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

TEST_FORCE_LINK(test_margin_container)

#include "scene/gui/control.h"
#include "scene/gui/margin_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestMarginContainer {

TEST_CASE("[SceneTree][MarginContainer] Default margins and layout") {
	MarginContainer *margin_container = memnew(MarginContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(margin_container);

	Control *child_control = memnew(Control);
	child_control->set_custom_minimum_size(Size2(20, 10));
	margin_container->add_child(child_control);

	margin_container->set_size(Size2(100, 80));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			margin_container->get_margin_size(SIDE_LEFT) == 0,
			"Left margin is 0 by default.");

	CHECK_MESSAGE(
			margin_container->get_margin_size(SIDE_TOP) == 0,
			"Top margin is 0 by default.");

	CHECK_MESSAGE(
			margin_container->get_margin_size(SIDE_RIGHT) == 0,
			"Right margin is 0 by default.");

	CHECK_MESSAGE(
			margin_container->get_margin_size(SIDE_BOTTOM) == 0,
			"Bottom margin is 0 by default.");

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(0, 0)),
			"Child control starts at the top-left corner when all margins are 0.");

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 80)),
			"Child control fills the container area when all margins are 0.");

	CHECK_MESSAGE(
			margin_container->get_minimum_size().is_equal_approx(Size2(20, 10)),
			"Minimum size equals the child minimum size when all margins are 0.");

	memdelete(child_control);
	memdelete(margin_container);
}

TEST_CASE("[SceneTree][MarginContainer] Custom margins affect layout and minimum size") {
	MarginContainer *margin_container = memnew(MarginContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(margin_container);

	margin_container->add_theme_constant_override("margin_left", 10);
	margin_container->add_theme_constant_override("margin_top", 20);
	margin_container->add_theme_constant_override("margin_right", 30);
	margin_container->add_theme_constant_override("margin_bottom", 40);

	Control *child_control = memnew(Control);
	child_control->set_custom_minimum_size(Size2(20, 10));
	margin_container->add_child(child_control);

	margin_container->set_size(Size2(200, 150));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(10, 20)),
			"Child control is offset by left and top margins.");

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(160, 90)),
			"Child control size is reduced by left/right and top/bottom margins.");

	CHECK_MESSAGE(
			margin_container->get_minimum_size().is_equal_approx(Size2(60, 70)),
			"Minimum size equals child minimum size plus all configured margins.");

	memdelete(child_control);
	memdelete(margin_container);
}

TEST_CASE("[SceneTree][MarginContainer] Multiple children use maximum minimum size") {
	MarginContainer *margin_container = memnew(MarginContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(margin_container);

	margin_container->add_theme_constant_override("margin_left", 1);
	margin_container->add_theme_constant_override("margin_top", 2);
	margin_container->add_theme_constant_override("margin_right", 3);
	margin_container->add_theme_constant_override("margin_bottom", 4);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(40, 10));
	child_control_2->set_custom_minimum_size(Size2(20, 50));
	margin_container->add_child(child_control_1);
	margin_container->add_child(child_control_2);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			margin_container->get_minimum_size().is_equal_approx(Size2(44, 56)),
			"Minimum size uses maximum child minimum width/height plus margins.");

	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(margin_container);
}

} // namespace TestMarginContainer

/**************************************************************************/
/*  test_center_container.cpp                                             */
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

TEST_FORCE_LINK(test_center_container)

#include "scene/gui/center_container.h"
#include "scene/gui/control.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestCenterContainer {

TEST_CASE("[SceneTree][CenterContainer] Standard behavior") {
	CenterContainer *center_container = memnew(CenterContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(center_container);
	Control *child_control = memnew(Control);
	center_container->add_child(child_control);

	center_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(50, 50)),
			"Child control is centered within the CenterContainer by default.");

	child_control->set_custom_minimum_size(Size2(20, 20));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(40, 40)),
			"Child control remains centered when custom minimum size is set.");

	child_control->set_custom_minimum_size(Size2(200, 200));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			center_container->get_size().is_equal_approx(Size2(200, 200)),
			"CenterContainer expands to accommodate the minimum size of its child control.");

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(0, 0)),
			"Child control is positioned at the top-left corner when its minimum size equals the container size.");

	child_control->set_custom_minimum_size(Size2());
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(50, 50)),
			"Child control re-centers when custom minimum size is reset.");

	CHECK_MESSAGE(
			center_container->get_size().is_equal_approx(Size2(100, 100)),
			"CenterContainer returns to original size when child control's custom minimum size is reset.");

	center_container->set_use_top_left(true);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(0, 0)),
			"Child control is aligned to the top-left corner when use_top_left is enabled.");

	center_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(100, 0)),
			"Child control is aligned to the top-right corner in RTL layout direction when use_top_left is enabled.");

	memdelete(child_control);
	memdelete(center_container);
}

TEST_CASE("[SceneTree][CenterContainer] Multiple children") {
	CenterContainer *center_container = memnew(CenterContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(center_container);

	center_container->set_size(Size2(100, 100));

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	center_container->add_child(child_control_1);
	center_container->add_child(child_control_2);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(50, 50)),
			"First child control is centered within the CenterContainer.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(50, 50)),
			"Second child control is also centered and overlaps the first child control by default.");

	child_control_1->set_custom_minimum_size(Size2(20, 20));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(40, 40)),
			"First child control remains centered when custom minimum size is set.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(50, 50)),
			"Second child control remains centered and overlaps the first child control even when the first child has a custom minimum size.");

	child_control_1->set_custom_minimum_size(Size2(200, 0));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			center_container->get_size().is_equal_approx(Size2(200, 100)),
			"CenterContainer expands to accommodate the minimum size of the first child control.");

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(0, 50)),
			"First child control is aligned to the left edge of the CenterContainer when its minimum width equals the container width.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(100, 50)),
			"Second child control is centered on the new container center.");

	child_control_2->set_custom_minimum_size(Size2(0, 200));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			center_container->get_size().is_equal_approx(Size2(200, 200)),
			"CenterContainer expands to accommodate the minimum size of the second child control.");

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(0, 100)),
			"First child control is aligned to the left edge and centered vertically when its minimum width equals the container width but its minimum height is smaller than the container height.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(100, 0)),
			"Second child control is aligned to the top edge and centered horizontally when its minimum height equals the container height but its minimum width is smaller than the container width.");

	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(center_container);
}

} // namespace TestCenterContainer

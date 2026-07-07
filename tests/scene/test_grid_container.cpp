/**************************************************************************/
/*  test_grid_container.cpp                                               */
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

TEST_FORCE_LINK(test_grid_container)

#include "scene/gui/control.h"
#include "scene/gui/grid_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestGridContainer {

TEST_CASE("[SceneTree][GridContainer] Default properties") {
	GridContainer *grid_container = memnew(GridContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(grid_container);

	grid_container->set_size(Size2(200, 200));
	grid_container->add_theme_constant_override("h_separation", 0);
	grid_container->add_theme_constant_override("v_separation", 0);

	Control *child_control = memnew(Control);
	child_control->set_custom_minimum_size(Size2(20, 10));
	grid_container->add_child(child_control);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			grid_container->get_columns() == 1,
			"GridContainer columns are set to 1 by default.");

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(0, 0)),
			"Single child control starts at the top-left corner.");

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(20, 10)),
			"Single child control keeps its minimum size when no expansion is requested.");

	memdelete(child_control);
	memdelete(grid_container);
}

TEST_CASE("[SceneTree][GridContainer] Columns and placement order") {
	GridContainer *grid_container = memnew(GridContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(grid_container);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	Control *child_control_3 = memnew(Control);
	Control *child_control_4 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(10, 10));
	child_control_2->set_custom_minimum_size(Size2(10, 10));
	child_control_3->set_custom_minimum_size(Size2(10, 10));
	child_control_4->set_custom_minimum_size(Size2(10, 10));

	grid_container->set_columns(2);
	grid_container->set_size(Size2(200, 200));
	grid_container->add_theme_constant_override("h_separation", 0);
	grid_container->add_theme_constant_override("v_separation", 0);
	grid_container->add_child(child_control_1);
	grid_container->add_child(child_control_2);
	grid_container->add_child(child_control_3);
	grid_container->add_child(child_control_4);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			grid_container->get_columns() == 2,
			"GridContainer uses the configured number of columns.");

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(0, 0)),
			"First child control is placed at row 0, column 0.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(10, 0)),
			"Second child control is placed at row 0, column 1.");

	CHECK_MESSAGE(
			child_control_3->get_position().is_equal_approx(Point2(0, 10)),
			"Third child control is placed at row 1, column 0.");

	CHECK_MESSAGE(
			child_control_4->get_position().is_equal_approx(Point2(10, 10)),
			"Fourth child control is placed at row 1, column 1.");

	memdelete(child_control_4);
	memdelete(child_control_3);
	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(grid_container);
}

TEST_CASE("[SceneTree][GridContainer] Separation and minimum size") {
	GridContainer *grid_container = memnew(GridContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(grid_container);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	Control *child_control_3 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(10, 20));
	child_control_2->set_custom_minimum_size(Size2(30, 10));
	child_control_3->set_custom_minimum_size(Size2(15, 40));

	grid_container->set_columns(2);
	grid_container->add_theme_constant_override("h_separation", 5);
	grid_container->add_theme_constant_override("v_separation", 7);
	grid_container->add_child(child_control_1);
	grid_container->add_child(child_control_2);
	grid_container->add_child(child_control_3);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			grid_container->get_h_separation() == 5,
			"GridContainer reports overridden horizontal separation.");

	CHECK_MESSAGE(
			grid_container->get_minimum_size().is_equal_approx(Size2(50, 67)),
			"GridContainer minimum size matches per-column and per-row maxima plus separations.");

	grid_container->set_size(Size2(50, 67));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(0, 0)),
			"First child control is placed at row 0, column 0 with configured separations.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(20, 0)),
			"Second child control is offset by column width and horizontal separation.");

	CHECK_MESSAGE(
			child_control_3->get_position().is_equal_approx(Point2(0, 27)),
			"Third child control is offset by row height and vertical separation.");

	CHECK_MESSAGE(
			child_control_1->get_size().is_equal_approx(Size2(15, 20)),
			"First child control fills the first grid cell dimensions.");

	CHECK_MESSAGE(
			child_control_2->get_size().is_equal_approx(Size2(30, 20)),
			"Second child control fills the second grid cell dimensions.");

	CHECK_MESSAGE(
			child_control_3->get_size().is_equal_approx(Size2(15, 40)),
			"Third child control fills the second-row first-column cell dimensions.");

	memdelete(child_control_3);
	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(grid_container);
}

TEST_CASE("[SceneTree][GridContainer] Expanding rows and columns") {
	GridContainer *grid_container = memnew(GridContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(grid_container);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	Control *child_control_3 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(20, 20));
	child_control_2->set_custom_minimum_size(Size2(20, 20));
	child_control_3->set_custom_minimum_size(Size2(20, 20));
	child_control_1->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	child_control_1->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	grid_container->set_columns(2);
	grid_container->add_theme_constant_override("h_separation", 0);
	grid_container->add_theme_constant_override("v_separation", 0);
	grid_container->set_size(Size2(100, 100));
	grid_container->add_child(child_control_1);
	grid_container->add_child(child_control_2);
	grid_container->add_child(child_control_3);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control_1->get_size().is_equal_approx(Size2(80, 80)),
			"Expanded child control grows with its expanded column and row.");

	CHECK_MESSAGE(
			child_control_2->get_size().is_equal_approx(Size2(20, 80)),
			"Non-expanded column still receives expanded row height.");

	CHECK_MESSAGE(
			child_control_3->get_size().is_equal_approx(Size2(80, 20)),
			"Non-expanded row still receives expanded column width.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(80, 0)),
			"Second child control is positioned after expanded first column width.");

	CHECK_MESSAGE(
			child_control_3->get_position().is_equal_approx(Point2(0, 80)),
			"Third child control is positioned after expanded first row height.");

	memdelete(child_control_3);
	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(grid_container);
}

TEST_CASE("[SceneTree][GridContainer] RTL placement") {
	GridContainer *grid_container = memnew(GridContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(grid_container);

	Control *child_control_1 = memnew(Control);
	Control *child_control_2 = memnew(Control);
	child_control_1->set_custom_minimum_size(Size2(20, 20));
	child_control_2->set_custom_minimum_size(Size2(20, 20));

	grid_container->set_columns(2);
	grid_container->add_theme_constant_override("h_separation", 0);
	grid_container->add_theme_constant_override("v_separation", 0);
	grid_container->set_size(Size2(40, 20));
	grid_container->add_child(child_control_1);
	grid_container->add_child(child_control_2);
	grid_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control_1->get_position().is_equal_approx(Point2(20, 0)),
			"First child control starts in the right column in RTL mode.");

	CHECK_MESSAGE(
			child_control_2->get_position().is_equal_approx(Point2(0, 0)),
			"Second child control is placed in the left column in RTL mode.");

	memdelete(child_control_2);
	memdelete(child_control_1);
	memdelete(grid_container);
}

} // namespace TestGridContainer
